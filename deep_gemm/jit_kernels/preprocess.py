import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div
from .sideaware import sideaware_info, sideaware_enabled

# C++ code templates
includes = ('"deep_gemm/reorder_b.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto kL2HashBits = {L2_HASH_BITS};
constexpr auto kL2Optimization = {L2_OPTIMIZATION};

// Make a templated type
using ReorderType = ReorderB<128, GemmType::{GEMM_TYPE}, kL2HashBits, kL2Optimization>;

// Launch kernel
auto tma_b_desc = ReorderType::make_2d_tma_b_desc(b, n, k, num_groups);
ReorderType::run(out_b, b, nullptr, tma_b_desc, n, k, stream, num_sms, num_groups);
"""

def preprocess_reorder_b(b: torch.Tensor, out_b: torch.Tensor) -> None:
    """
    Reorder B tensor to match the smem layout and be L2 side aware(!)
    """
    n, k = b.shape
    assert b.shape == out_b.shape
    assert b.is_contiguous() and out_b.is_contiguous()
    assert b.dtype == torch.float8_e4m3fn and out_b.dtype == torch.float8_e4m3fn
    assert k % 128 == 0

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    args = (b, out_b, n, k, torch.cuda.current_stream(), num_sms, 1)
    runtime = jit_tuner.compile_and_tune(
        name='reorder_b',
        keys={'BLOCK_K': 128, 'GEMM_TYPE': 'Normal',
              'L2_HASH_BITS': sideaware_info()["hash"], 'L2_OPTIMIZATION': sideaware_enabled()},
        space=(),
        includes=includes,
        arg_defs=(('b', torch.float8_e4m3fn), ('out_b', torch.float8_e4m3fn),
                  ('n', int), ('k', int), ('stream', torch.cuda.Stream), ('num_sms', int), ('num_groups', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)

def preprocess_reorder_b_grouped(b: torch.Tensor, out_b: torch.Tensor, is_masked: bool) -> None:
    """
    Reorder B tensor to match the smem layout and be L2 side aware(!)
    """
    num_groups, n, k = b.shape
    assert b.shape == out_b.shape
    assert b.is_contiguous() and out_b.is_contiguous()
    assert b.dtype == torch.float8_e4m3fn and out_b.dtype == torch.float8_e4m3fn
    assert k % 128 == 0

    # Auto-tuning with compilation
    global includes, template
    num_sms = get_num_sms()
    args = (b, out_b, n, k, torch.cuda.current_stream(), num_sms, num_groups)
    runtime = jit_tuner.compile_and_tune(
        name='reorder_b',
        keys={'BLOCK_K': 128, 'GEMM_TYPE': 'GroupedContiguous' if not is_masked else 'GroupedMasked',
              'L2_HASH_BITS': sideaware_info()["hash"], 'L2_OPTIMIZATION': sideaware_enabled()},
        space=(),
        includes=includes,
        arg_defs=(('b', torch.float8_e4m3fn), ('out_b', torch.float8_e4m3fn),
                  ('n', int), ('k', int), ('stream', torch.cuda.Stream), ('num_sms', int), ('num_groups', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)
