import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_col_major_tma_aligned_tensor, get_m_alignment_for_contiguous_layout
from .sideaware import sideaware_torch_side_index, sideaware_info, sideaware_enabled

# C++ code templates
includes = ('"deep_gemm/fp8_gemm.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumUnroll = {NUM_UNROLL};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};
constexpr auto kL2HashBits = {L2_HASH_BITS};
constexpr auto kL2Optimization = {L2_OPTIMIZATION};

// Make a templated GEMM
using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, 1, kNumStages, kNumUnroll, kNumTMAMulticast, GemmType::Normal, kL2HashBits, kL2Optimization>;

// Launch kernel
auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, m);
auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs);
auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales, m);
auto tma_d_desc = GemmType::make_2d_tma_d_desc(out, m);
auto tma_d_padded_desc = GemmType::make_3d_tma_d_desc(out, m);
GemmType::run(out, rhs, rhs_scales, nullptr,
              m,
              tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc, tma_d_padded_desc,
              stream, num_sms, smem_size, side_index);
"""


def is_tma_multicast_legal(n: int, block_n: int, num_tma_multicast: int, num_sms: int) -> bool:
    if num_tma_multicast == 1:
        return True
    return (n % (block_n * num_tma_multicast) == 0) and num_sms % num_tma_multicast == 0


def get_smem_size(num_stages: int, k: int, block_m: int, block_n: int, block_k: int = 128) -> int:
    smem_d = block_m * block_n * 2
    smem_a_per_stage = block_m * block_k
    smem_scales_a_per_stage = block_m * 4
    smem_b_per_stage = block_n * block_k
    smem_scales_b = ceil_div(k, block_k) * 4 * 2
    smem_barrier = (num_stages + 2) * 8 * 2

    # D reuses AB pipeline stage but potentially larger due to padding to avoid bank conflicts
    assert(smem_d % 128 == 0) # 128 because of other alignment considerations elsewhere
    PADDING_N = (block_n == 64 or block_n == 96 or block_n == 128) and 16 or 0
    smem_d_padded = block_m * (block_n + PADDING_N) * 2
    smem_ab_per_stage = max(smem_a_per_stage + smem_b_per_stage, smem_d_padded)

    smem_size = 0
    # we reuse one stage of A+B to store D instead of dedicated storage
    # smem_size += smem_d
    assert smem_d <= (smem_a_per_stage + smem_b_per_stage)
    smem_size += num_stages * smem_ab_per_stage
    smem_size += num_stages * smem_scales_a_per_stage
    smem_size += ceil_div(smem_scales_b * (1 if block_k % block_n == 0 else 2), 8) * 8
    smem_size += smem_barrier
    smem_size += 4096  # scratch for tile scheduling etc.
    return smem_size


def get_best_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                     is_grouped_contiguous: bool = False) -> Tuple[int, int, int, int, int]:
    if not is_grouped_contiguous:
        # TODO: for some cases, smaller M block is better, add them into tuning space
        block_ms = (64 if m <= 64 else 128, )
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(), )
    # TODO: add back other sizes if we switch to TMA tensor loads instead, multiples of 8 are slow in current approach
    block_ns = tuple((16, 32, 48, 64, 96, 128)) #range(16, 129, 8))

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        for block_n in block_ns:
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            if best_block_m is None or best_block_n is None:
                success = True
            elif num_waves < best_num_waves:
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                util = get_last_wave_util(block_m, block_n)
                best_util = get_last_wave_util(best_block_m, best_block_n)
                success = util > best_util or (util == best_util and (block_m > best_block_m or (block_m == best_block_m and block_n < best_block_n)))
            best_block_m, best_block_n = (block_m, block_n) if success else (best_block_m, best_block_n)
    assert best_block_m is not None and best_block_n is not None

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_size, sm90_capacity = None, None, 232448
    # TODO: what sizes to check?
    for num_stages in (9, 8, 7, 6, 5, 4) if 128 % best_block_n != 0 else (12, 10, 8, 7, 6, 5, 4):
        best_smem_size = get_smem_size(num_stages, k, best_block_m, best_block_n)
        if best_smem_size <= sm90_capacity:
            best_num_stages = num_stages
            break
    assert best_num_stages is not None

    # Decide the number of TMA multicast
    best_num_tma_multicast = 1
    if m >= 1024 and is_tma_multicast_legal(n, best_block_n, 2, num_sms) and (num_groups == 1 or is_grouped_contiguous):
        best_num_tma_multicast = 2


    if False:
        print(f"best_block_m: {best_block_m}, best_block_n: {best_block_n}, best_num_stages: {best_num_stages},"
            f"best_smem_size: {best_smem_size}, best_num_tma_multicast: {best_num_tma_multicast}, m: {m}, n: {n}, k: {k}"
            f"==> Waves: {get_num_waves(best_block_m, best_block_n)}")


    return best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size


def gemm_fp8_fp8_bf16_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                         rhs: Tuple[torch.Tensor, torch.Tensor],
                         out: torch.Tensor) -> None:
    """
    Do a normal GEMM with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m, n]`, representing the result.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    assert n % 64 == 0 and k % 128 == 0

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == ((n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, n, k, 1, num_sms)
    args = (lhs, lhs_scales, rhs, rhs_scales, out, m, torch.cuda.current_stream(), num_sms, smem_size,
            sideaware_torch_side_index())
    runtime = jit_tuner.compile_and_tune(
        name='gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast,
              'L2_HASH_BITS': sideaware_info()["hash"], 'L2_OPTIMIZATION': sideaware_enabled()},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int),
                  ('side_index', torch.uint8)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)
