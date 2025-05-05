

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm {

// TODO - these settings shouldn't be here.
constexpr bool DOUBLE_PUMP = true; // todo - figure out how we can make this *always* faster (not just usually so...)
constexpr bool DP_SCALE_256 = true; // todo - assumes A/B scales are always the same for 2 blocks, need test data here

constexpr int MAX_SM = 132;
constexpr int PADDING_N = 16; // padding for D to avoid STSM bank conflicts (todo - clearer conditions etc.)
constexpr int NUM_TILES_INITIAL = 32; // calxulate m/n for INITIAL tiles in parallel in prologue
constexpr int NUM_TILES_STORAGE = 64; // 1 more every time we load B scales in a round-robin buffer

enum class Layout {
    RowMajor,
    ColMajor
};

template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return (block_m == 64 ? 1 : 2) * kNumMathThreadsPerGroup + kNumTMAThreads;
}

typedef struct {
    uint8_t sm_side_and_idx[MAX_SM];
} param_side_index_t;

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages, uint32_t kNumUnroll,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          bool     kTMAMulticastEnabled, GemmType kGemmType, uint32_t l2_hash_bits, bool l2_optimization,
          uint32_t FORCED_M>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel(__nv_bfloat16* gmem_d, __nv_fp8_e4m3* gmem_b, float* scales_b, int* grouped_layout, int* zeroed_scratch,
                uint32_t shape_m,
                const __grid_constant__ CUtensorMap tensor_map_a,
                const __grid_constant__ CUtensorMap tensor_map_b,
                const __grid_constant__ CUtensorMap tensor_map_scales_a,
                const __grid_constant__ CUtensorMap tensor_map_d,
                const __grid_constant__ CUtensorMap tensor_map_d_padded,
                int block_idx_base, int aggregate_grid_size,
                __grid_constant__ const param_side_index_t sideaware) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    //
    constexpr bool L2_SIDE_OPTIMIZATION = l2_optimization && (kGemmType != GemmType::GroupedMasked);
    constexpr uint32_t SHAPE_N_HALF = SHAPE_N / 2;
    constexpr uint32_t CLUSTER_BLOCK_N = BLOCK_N * (kTMAMulticastEnabled ? 2 : 1);
    constexpr uint32_t SHAPE_N_LOWER = ((SHAPE_N_HALF + CLUSTER_BLOCK_N - 1) / CLUSTER_BLOCK_N) * CLUSTER_BLOCK_N;
    constexpr uint32_t SHAPE_N_UPPER = SHAPE_N - SHAPE_N_LOWER;
    constexpr uint32_t SHAPE_N_MAX = L2_SIDE_OPTIMIZATION ? SHAPE_N_LOWER : SHAPE_N;

    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_N, BLOCK_K) == 1, "Too much B scales in a single block");

    // The compiler can optionally optimize everything if shape_m is known at compile time
    shape_m = FORCED_M ? FORCED_M : shape_m;
    // TODO: Re-enable hybrid cluster support
    //if constexpr (!kTMAMulticastEnabled) {
        //aggregate_grid_size = gridDim.x;
        block_idx_base = 0;
    //}

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    constexpr uint32_t PADDING = (BLOCK_N == 64 || BLOCK_N == 96 || BLOCK_N == 128) ? PADDING_N : 0;
    constexpr uint32_t BLOCK_N_PADDED = BLOCK_N + PADDING;

    // Shared memory
    static constexpr int kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_D_SIZE_PADDED = BLOCK_M * BLOCK_N_PADDED * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_AB_SIZE_PER_STAGE_RAW = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
    static constexpr uint32_t SMEM_AB_SIZE_PER_STAGE = SMEM_D_SIZE_PADDED > SMEM_AB_SIZE_PER_STAGE_RAW ? SMEM_D_SIZE_PADDED : SMEM_AB_SIZE_PER_STAGE_RAW;
    static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_B_SIZE = ceil_div<uint32_t>(SHAPE_K_SCALES * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Fill shared memory *base* pointers
    // Everything is contiguous in memory, so we can do address calculations instead storing loads of pointers
    // this is needed for performance without fully unrolling the loops (otherwise we see LDL/STL for array indexing)
    // A and B are interleaved (A[0], B[0], A[1], B[1], ...) so they can be reused for D, everything else is contiguous
    __nv_fp8_e4m3* smem_a_base = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer);
    __nv_fp8_e4m3* smem_b_base = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_A_SIZE_PER_STAGE);
    float* smem_scales_a_base = reinterpret_cast<float*>(smem_buffer + kNumStages * (SMEM_AB_SIZE_PER_STAGE));
    float* smem_scales_b_base = reinterpret_cast<float*>(smem_buffer + kNumStages * (SMEM_AB_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE));
    constexpr int kNumScalesB = SHAPE_K_SCALES * (kMustUseUniformedScaleB ? 1 : 2);

    // Fill barriers (base pointers only)
    DG_STATIC_ASSERT(sizeof(Barrier) % sizeof(float) == 0, "Misaligned barriers");
    DG_STATIC_ASSERT(not kMustUseUniformedScaleB or SHAPE_K_SCALES % (sizeof(Barrier) / sizeof(float)) == 0, "Misaligned barriers");
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_b_base) + (2*SMEM_SCALES_B_SIZE));
    auto full_barriers_base = barrier_start_ptr;
    auto empty_barriers_base = barrier_start_ptr + kNumStages;
    auto full_barrier_scales_b_base = barrier_start_ptr + kNumStages * 2;
    auto empty_barrier_scales_b_base = barrier_start_ptr + kNumStages * 2 + 2; // double-buffered
    uint4* smem_tile_scheduling = reinterpret_cast<uint4*>(empty_barrier_scales_b_base + 2);

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();
    uint32_t cluster_size; // get cluster_nctaid
    asm volatile("mov.u32 %0, %cluster_nctaid.x;" : "=r"(cluster_size));

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, SHAPE_N_MAX, BLOCK_M, BLOCK_N, kNumGroups>
                            (shape_m, grouped_layout, block_idx_base + blockIdx.x);


    if constexpr (L2_SIDE_OPTIMIZATION) {
        int smid;
        asm("mov.u32 %0, %smid;\n" : "=r"(smid) :);
        int side = sideaware.sm_side_and_idx[smid] & 1;
        scheduler.block_idx = sideaware.sm_side_and_idx[smid] >> 1;
        scheduler.n_block_offset = !side * (SHAPE_N_LOWER / BLOCK_N);
        scheduler.grid_size /= 2;
    }

    // Pre-compute tile m/n for NUM_TILES_INITIAL tiles during warmup
    // Future tiles will be computed in the same thread that loads B scales (both are per-tile rather than per-block)
    if (threadIdx.x < NUM_TILES_INITIAL) {
        if (threadIdx.x*scheduler.grid_size < scheduler.num_blocks+scheduler.grid_size || kGemmType == GemmType::GroupedMasked) {
            scheduler.current_iter = threadIdx.x - 1;
            scheduler.get_next_block(m_block_idx, n_block_idx);
            smem_tile_scheduling[threadIdx.x].x = m_block_idx;
            smem_tile_scheduling[threadIdx.x].y = n_block_idx;
            if constexpr (kGemmType == GemmType::GroupedMasked) {
                smem_tile_scheduling[threadIdx.x].z = scheduler.curr_group_idx;
                smem_tile_scheduling[threadIdx.x].w = 0;
            }
            scheduler.current_iter = -1;
        }
    }

    // Helper lambda to fetch next tile from shared memory (precomputed way in advance)
    // this only works because we use static scheduling without work stealing, which has its own benefits...
    auto fetch_next_tile = [&](uint32_t& m_block_idx, uint32_t& n_block_idx) -> bool {
        scheduler.current_iter++;
        int idx = scheduler.current_iter % NUM_TILES_STORAGE;

        m_block_idx = smem_tile_scheduling[idx].x;
        n_block_idx = smem_tile_scheduling[idx].y;
        if constexpr (kGemmType == GemmType::GroupedMasked) {
            scheduler.curr_group_idx = smem_tile_scheduling[idx].z;
        }
        return (m_block_idx != 0xFFFFFFFF);
    };

    // Initialize barriers (split over threads, it can be a lot with 10+ stages!)
    // Keep threads associated with 1st sub-processor (threads 0 to 31) free since they do other work above
    if (threadIdx.x == kNumMathThreads + 32) {
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            full_barriers_base[i].init(2);
        }
        cutlass::arch::fence_view_async_shared();
        (kTMAMulticastEnabled) ? cutlass::arch::fence_barrier_init() : void();
    } else if (threadIdx.x == kNumMathThreads + 64) {
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            empty_barriers_base[i].init(cluster_size * kNumMathThreads / 32);
        }
        cutlass::arch::fence_view_async_shared();
        (kTMAMulticastEnabled) ? cutlass::arch::fence_barrier_init() : void();
    } else if (threadIdx.x == kNumMathThreads + 96) {
         // Prefetch TMA descriptors at very beginning
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d_padded));
        #pragma unroll
        for (int i = 0; i < 2; ++ i) {
            full_barrier_scales_b_base[i].init(1);
            empty_barrier_scales_b_base[i].init(kNumMathThreads / 32);
        }
        cutlass::arch::fence_view_async_shared();
        (kTMAMulticastEnabled) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Register reconfigurations
    // TODO - this was to allow 256 'TMA' threads, but no apparent benefit from more registers than this anyway
    constexpr int kNumTMARegisters = 32;
    constexpr int kNumMathRegisters = 224;

    // Synchronize all threads to make barrier visible in normal memory model (as late as possible)
    (kTMAMulticastEnabled) ? cute::cluster_sync() : __syncthreads();

    // Updates pipeline stage & parity in as few SASS instructions as possible
    int s = 0, last_s = -1, parity = -1; // persistent context across loop iterations
    auto next_stage = [&]() {
        bool wrap = (s == kNumStages-1);
        last_s = s;
        s++;
        if (wrap) {
            s = 0;
            parity++;
        }
    };

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-groups for loading data - we split the threads into
        // 1) Calculating future tile m/n and loading B scales per tile (1 thread)
        // 2) Loading A data & scales (1 thread)
        // 3) Loading B data (multiple threads/warps, expensive due to L2 side awareness, optimized PTX)
        //
        // (3) previously supported multiple warps to parallelise the L2 side calculations...
        // but slower after other crazy optimizations so back to 1 warp (but multiple threads per warp)

        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        parity = 1; // producer starts with parity=1 (no wait)

        if (warp_idx == kNumMathWarps) {
            // TODO - explain this code, or better yet, rewrite all of it!
            // TODO - add back "fast path" when everything is aligned, it was much faster *sigh* (or rewrite the whole thing!!!)
            constexpr int CHUNK_SIZE = (BLOCK_N % 16) ? 8 : ((BLOCK_N % 32) ? 16 : 32); // largest chunk size that can divide BLOCK_N
            constexpr int NUM_CHUNKS = (BLOCK_N + CHUNK_SIZE - 1) / CHUNK_SIZE;
            if constexpr (L2_SIDE_OPTIMIZATION && NUM_CHUNKS > 1) {
                if (lane_idx >= NUM_CHUNKS) {
                    return;
                }
            } else {
                elect_or_exit();
            }

            // Create a lambda to handle loading B data with different starting k_idx
            int loader_idx = 0;
            int loader_tid = threadIdx.x-kNumMathThreads;

            constexpr bool aligned_n = (SHAPE_N % BLOCK_N == 0 && SHAPE_N_LOWER % BLOCK_N == 0 && BLOCK_N % CHUNK_SIZE == 0);

            if constexpr (L2_SIDE_OPTIMIZATION) {
                int current_shape_n = (scheduler.n_block_offset) ? SHAPE_N : SHAPE_N_LOWER;
                int start_page_offset = reinterpret_cast<uint64_t>(gmem_b) % (2048*1024);

                __nv_fp8_e4m3* b_page_start = gmem_b - start_page_offset;
                uintptr_t b_page_start_u64 = reinterpret_cast<uintptr_t>(b_page_start);
                uint32_t b_page_start_u32[2];
                b_page_start_u32[0] = (uint32_t)(b_page_start_u64 & 0xFFFFFFFF);
                b_page_start_u32[1] = (uint32_t)(b_page_start_u64 >> 32L);

                smem_b_base += (NUM_CHUNKS > 1) ? lane_idx * (CHUNK_SIZE * BLOCK_K) : 0;

                int global_base_offset = start_page_offset;
                int lane_chunk_start = (NUM_CHUNKS > 1) ? (lane_idx * CHUNK_SIZE) : 0;

                // Persistently schedule over blocks to load B
                #pragma unroll 1
                while (fetch_next_tile(m_block_idx, n_block_idx)) {
                    int n = n_block_idx * BLOCK_N;

                    int remaining_n = current_shape_n - n;
                    n += lane_chunk_start;
                    int n_side = (n >= SHAPE_N_HALF) ? 1 : 0;
                    int n_half = (n_side * (-SHAPE_N_HALF)) + n;
                    int n_dst_base = n_half + (n_half & ~31); // shift everything after bit 5 to the left by 1
                    uint32_t tile_base_offset = global_base_offset + (n_dst_base * 128);
                    if constexpr (kGemmType == GemmType::GroupedContiguous) {
                        int group_offset = __ldg(grouped_layout + m_block_idx * BLOCK_M);
                        tile_base_offset += (SHAPE_N * SHAPE_K) * group_offset;
                    }

                    // Declare early so we can use a lambda to compile 2 optimized paths based on their values
                    int num_bytes_total;
                    int num_bytes;

                    // ----------------------------------------------------------------------------------------
                    // Check sideaware_kernel.cuh for a simpler version of the side aware algorithm without PTX
                    // ----------------------------------------------------------------------------------------
                    // Custom PTX implementation of side-aware memory copy for B matrix
                    // optimized for efficient SASS at a random driver in time (12.8)
                    // ... because why not?
                    // ----------------------------------------------------------------------------------------
                    auto load_b_for_every_k = [&]() {
                        #pragma unroll 1
                        for (int k_idx = 0; k_idx < SHAPE_K_SCALES; k_idx++, tile_base_offset += BLOCK_K * SHAPE_N) {
                            uint32_t smem_int_mbar = cute::cast_smem_ptr_to_uint(&full_barriers_base[s]);
                            uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(smem_b_base + s * SMEM_AB_SIZE_PER_STAGE);
                            uint32_t smem_int_empty_mbar = cute::cast_smem_ptr_to_uint(&empty_barriers_base[s]);
                            uint32_t gmem_address_u32[2];

                            // wait on mbarrier (can't remember if this ended up any better than the baseline barrier.wait)
                            asm volatile(
                                "{\n"
                                "    .reg .pred p;\n"
                                "    LAB_WAIT:\n"
                                "    mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, 10000000;\n"
                                "    @p bra DONE;\n"
                                "    bra LAB_WAIT;\n"
                                "    DONE:\n"
                                "}\n"
                                : : "r"(smem_int_empty_mbar), "r"(parity) : "memory"
                            );

                            // optimized PTX version of next_stage()
                            // this was required because we had multiple warps interleaving iterations of the loop
                            // not sure if this is actually faster than the baseline version with a single warp or not
                            asm volatile(
                                "{\n"
                                "    .reg .pred p;\n"
                                "    setp.ge.s32 p, %0, %2;\n"
                                "    @p add.s32 %0, %0, %3;\n"
                                "    @p add.s32 %1, %1, 1;\n"
                                "    @!p add.s32 %0, %0, 1;\n"
                                "}\n"
                                : "+r"(s), "+r"(parity)
                                : "n"(kNumStages - 1), "n"(1 - kNumStages)
                                : "memory"
                            );

                            if (num_bytes > 0) {
                                // Check sideaware_kernel.cuh for a simpler version of the side aware algorithm without PTX
                                //
                                // Determine the desired side based on the l2_hash_bits of the address (using popc)
                                // then use this to adjust the offset as required (see sideaware_kernel.cuh)
                                // the black magic part of this PTX is related to how we handle the unaligned case
                                // more efficiently than in my other non-PTX version of the algorithm
                                // it uses 'add.cc' and 'addc.u32' in a clever way to save a few instructions
                                asm volatile(
                                    "{\n"
                                    "    .reg .u32 lower_bits;\n"
                                    "    .reg .u32 tmp;\n"
                                    "    .reg .b64 address;\n"
                                    "    add.cc.u32 lower_bits, %2, %4;\n"
                                    "    and.b32 tmp, lower_bits, %5;\n"
                                    "    xor.b32 tmp, tmp, %6;\n"
                                    "    popc.b32 tmp, tmp;\n"
                                    "    and.b32 tmp, tmp, 0x1;\n"
                                    "    xor.b32 tmp, tmp, 0x1;\n" // invert due to bit 21 in sideaware.cu hash
                                    "    mad.lo.u32 %0, tmp, 4096, lower_bits;\n"
                                    "    addc.u32 %1, %3, 0;\n"
                                    : "=r"(gmem_address_u32[0])/*0*/,  "=r"(gmem_address_u32[1])/*1*/
                                    : "r"(b_page_start_u32[0]) /*2*/,  "r"(b_page_start_u32[1]) /*3*/,
                                    "r"(tile_base_offset)      /*4*/,  "r"(l2_hash_bits)        /*5*/, "r"(n_side) /*6*/
                                    : "memory"
                                );

                                if constexpr (NUM_CHUNKS == 1) {
                                    // Fast path with a single thread active per warp
                                    asm volatile(
                                        "    mov.b64 address, {%0, %1};\n"
                                        "    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%2], [address], %3, [%4];\n"
                                        "    mbarrier.arrive.expect_tx.shared::cta.b64 _, [%4], %5; \n"
                                        "}\n"
                                        : : "r"(gmem_address_u32[0]), "r"(gmem_address_u32[1]),
                                            "r"(smem_int_ptr), "r"(num_bytes), "r"(smem_int_mbar), "r"(num_bytes_total)
                                        : "memory"
                                    );
                                } else {
                                    // Slow path with multiple threads where the compiler will create a loop for the TMA
                                    // the SASS isn't optimal but extremely difficult to improve further with 'just' PTX
                                    // (+ setp/@p to only mbarrier.arrive on a single thread)
                                    asm volatile(
                                        "    .reg .pred p;\n"
                                        "    setp.eq.u32 p, %6, 0;\n"
                                        "    mov.b64 address, {%0, %1};\n"
                                        "    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%2], [address], %3, [%4];\n"
                                        "@p  mbarrier.arrive.expect_tx.shared::cta.b64 _, [%4], %5; \n"
                                        "}\n"
                                        : : "r"(gmem_address_u32[0]), "r"(gmem_address_u32[1]),
                                            "r"(smem_int_ptr), "r"(num_bytes), "r"(smem_int_mbar), "r"(num_bytes_total),
                                            "r"(lane_idx)
                                        : "memory"
                                    );
                                }
                            }
                        }
                    };

                    if (!aligned_n && remaining_n < BLOCK_N) {
                        // Slow path with dynamic num_byte
                        int n_to_load_warp = max(0, min(remaining_n, BLOCK_N));
                        int n_to_load_lane = (n_to_load_warp - lane_chunk_start);

                        num_bytes_total = n_to_load_warp * BLOCK_K;
                        num_bytes = (n_to_load_lane > CHUNK_SIZE) ? (CHUNK_SIZE*BLOCK_K) : (n_to_load_lane*BLOCK_K);
                        load_b_for_every_k();
                    } else {
                        // Fast path where the compiler realises num_bytes is known at compile time
                        num_bytes_total = BLOCK_N * BLOCK_K;
                        num_bytes = CHUNK_SIZE * BLOCK_K;
                        load_b_for_every_k();
                    }
                }
            } else {
                // Legacy approach without L2 side optimization
                while (fetch_next_tile(m_block_idx, n_block_idx)) {
                    for (int k_idx = 0; k_idx < SHAPE_K_SCALES; k_idx++) {
                        auto& full_barrier = full_barriers_base[s];
                        uint64_t* full_barrier64 = reinterpret_cast<uint64_t*>(&full_barrier);
                        empty_barriers_base[s].wait(parity);
                        tma_copy(&tensor_map_b, full_barrier64, smem_b_base + s * SMEM_AB_SIZE_PER_STAGE,
                                k_idx * BLOCK_K, scheduler.get_global_idx<false>(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx), cluster_size);
                        full_barriers_base[s].arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                        s++;
                        if (s >= kNumStages) {
                            s -= kNumStages;
                            parity++;
                        }
                    }
                }
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kTMAMulticastEnabled) {
                if (loader_idx == 0 && lane_idx == 0) {
                    for (int i = 0; i < kNumStages+1; i++) {
                        empty_barriers_base[s].wait(parity);
                        next_stage();
                    }
                }
            }
        } else if (threadIdx.x == kNumMathThreads + 64) {
            elect_or_exit(); // tell nvcc this is single-threaded (bad codegen otherwise on 12.8)
            // Persistently schedule over blocks to load A/A_scales
            while (fetch_next_tile(m_block_idx, n_block_idx)) {
                if (!kTMAMulticastEnabled || cute::block_rank_in_cluster() == 0) {
                    #pragma unroll 2 // TODO: only if divisible by 2
                    for (int k_idx = 0; k_idx < SHAPE_K_SCALES; k_idx++) {
                        // Wait consumer release
                        empty_barriers_base[s].wait(parity);
                        // Issue TMA A with broadcasting
                        auto& full_barrier = full_barriers_base[s];
                        uint64_t* full_barrier64 = reinterpret_cast<uint64_t*>(&full_barrier);
                        tma_copy<kTMAMulticastEnabled>(&tensor_map_a, full_barrier64, smem_a_base + s * SMEM_AB_SIZE_PER_STAGE,
                                                       k_idx * BLOCK_K, scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx), cluster_size);
                        tma_copy<kTMAMulticastEnabled>(&tensor_map_scales_a, full_barrier64, smem_scales_a_base + s * BLOCK_M,
                                                       m_block_idx * BLOCK_M, scheduler.get_global_idx(SHAPE_K_SCALES, 1, k_idx), cluster_size);
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                        next_stage();
                    }
                } else {
                    // "block_rank_in_cluster() != 0" only need to release barriers at the right time, no TMAs needed
                    #pragma unroll 1
                    for (int k_idx = 0; k_idx < SHAPE_K_SCALES; k_idx++) {
                        empty_barriers_base[s].wait(parity);
                        full_barriers_base[s].arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                        next_stage();
                    }
                }
            }
            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kTMAMulticastEnabled) {
                for (int i = 0; i < kNumStages + 1; i++) {
                    empty_barriers_base[s].wait(parity);
                    next_stage();
                }
            }
        } else if (threadIdx.x == kNumMathThreads + 96) {
            // Load B Scales per-tile + future tile scheduling (many tiles in advance)
            elect_or_exit();
            auto future_scheduler = scheduler; // deep copy of scheduler
            future_scheduler.current_iter = NUM_TILES_INITIAL - 1;

            // Load scales B via TMA Load per-tile rather than per-K_BLOCK
            // previously done with global memory loads, which was OK but forced synchronization between warpgroups
            // hardcoded to always be doubled-buffered (equivalent to s=2) which is more than enough (but s=1 isn't!)
            #pragma unroll 1
            while (fetch_next_tile(m_block_idx, n_block_idx)) {
                auto num_previous_lines = scheduler.get_global_idx<false>(ceil_div(SHAPE_N, BLOCK_K), 0, 0, m_block_idx);
                auto local_scales_b = scales_b + (num_previous_lines + ((n_block_idx * BLOCK_N) / BLOCK_K)) * SHAPE_K_SCALES;

                // Decide the number of scales B to load
                // this is inside the loop because it's index-dependent for N != 128 and non-uniform ScaleB
                DG_STATIC_ASSERT(SHAPE_N % 8 == 0, "Invalid shape N");
                uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
                if constexpr (not kMustUseUniformedScaleB) {
                    num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                    num_full_iters = min(SHAPE_N - n_block_idx * BLOCK_N, BLOCK_N) / 8;
                }
                uint32_t num_scales_b = SHAPE_K_SCALES * (num_former_iters >= num_full_iters ? 1 : 2);

                // explicit idx/parity calculation since (same or fewer instructions than increment+wrap)
                int barrier_idx = scheduler.current_iter & 1;
                int scales_parity = (scheduler.current_iter & 2) ? 0 : 1; // init=1 for producer (0 for consumer)
                empty_barrier_scales_b_base[barrier_idx].wait(scales_parity);

                auto& full_barrier = full_barrier_scales_b_base[barrier_idx];
                cute::SM90_BULK_COPY_G2S::copy(local_scales_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                               smem_scales_b_base + barrier_idx * kNumScalesB, num_scales_b * sizeof(float));
                full_barrier.arrive_and_expect_tx(num_scales_b * sizeof(float));

                ////// TODO - explain future tile scheduling, currently single threaded with implicit synchronization
                DG_STATIC_ASSERT(NUM_TILES_INITIAL > kNumStages+8, "NUM_TILES_INITIAL should be much greater than kNumStages");
                uint32_t future_m_block_idx, future_n_block_idx;
                future_scheduler.get_next_block(future_m_block_idx, future_n_block_idx);

                int tile_smem_idx = future_scheduler.current_iter % NUM_TILES_STORAGE;
                smem_tile_scheduling[tile_smem_idx].x = future_m_block_idx;
                smem_tile_scheduling[tile_smem_idx].y = future_n_block_idx;
                if constexpr (kGemmType == GemmType::GroupedMasked) {
                    smem_tile_scheduling[tile_smem_idx].z = future_scheduler.curr_group_idx;
                    smem_tile_scheduling[tile_smem_idx].w = 0;
                }
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        parity = 0; // consumer starts with parity=0 (must wait)

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        // Accumulation for WGMMA or CUDA promotion
        // We use 2 temporary accumulators for WGMMAs and one final accumulator for CUDA promotion
        float accum0[WGMMA::kNumAccum], accum1[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};
        __nv_bfloat162 final_accum_bf16[WGMMA::kNumAccum / 2];

        // The descriptors are basically always the same plus an offset, so we precompute them here
        uint64_t desc_a_base = make_smem_desc(smem_a_base + math_wg_idx * WGMMA::M * BLOCK_K, 1);
        uint64_t desc_b_base = make_smem_desc(smem_b_base, 1);

        // Empty barrier arrival
        auto empty_barrier_arrive = [&](Barrier* barrier) {
            if constexpr (!kTMAMulticastEnabled) {
                lane_idx == 0 ? barrier->arrive() : void();
            } else {
                lane_idx < cluster_size ? barrier->arrive(lane_idx) : void();
            }
        };

        // Keep track of the previous tile's position to do its TMA store in the next loop iteration
        uint32_t old_global_idx;
        int old_n_block_idx = -1;

        //---------------------------------------------------------------------------------
        // Lambda to store tile N-1 in iteration N and the final tile after the loop
        // final_accum (64 registers for N=128) ==> SMEM via STSM ==> global via TMA
        //---------------------------------------------------------------------------------
        auto final_accum_to_bf16 = [&]() {
            for (int i = 0; i < WGMMA::kNumAccum / 2; ++ i) {
                final_accum_bf16[i] = __float22bfloat162_rn({final_accum[i*2+0], final_accum[i*2+1]});
            }
        };
        auto store_tile = [&] (int tile_s, int start=0, int end=WGMMA::kNumAccum, bool skip_to_bf16=false) {
            int current_shape_n = (scheduler.n_block_offset) ? SHAPE_N : SHAPE_N_LOWER;

            // Write final_accum to shared memory using STSM
            // Padded to avoid up to 8x(!) shared memory bank conflicts
            auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_a_base + tile_s * (SMEM_AB_SIZE_PER_STAGE));
            bool partially_oob = (old_n_block_idx * BLOCK_N) > (SHAPE_N - BLOCK_N);
            uint32_t BLOCK_N_STORE = partially_oob ? BLOCK_N : BLOCK_N_PADDED;

            // Only process part of the tile at a time if possible
            if (start == 0 && !skip_to_bf16) final_accum_to_bf16();

            // Write back to shared memory using STSM
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (auto i = start / 8; i < end / 8; ++ i) {
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    final_accum_bf16[i * 4 + 0], final_accum_bf16[i * 4 + 1],
                    final_accum_bf16[i * 4 + 2], final_accum_bf16[i * 4 + 3],
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N_STORE + i * 16 + 8 * (lane_idx / 16)
                );
            }

            // TMA store on final iteration
            if (end >= WGMMA::kNumAccum && start < WGMMA::kNumAccum) {
                if constexpr (WGMMA::kNumAccum % 8 != 0) {
                    SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        final_accum_bf16[WGMMA::kNumAccum / 8 * 4 + 0], final_accum_bf16[WGMMA::kNumAccum / 8 * 4 + 1],
                        smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N_STORE + WGMMA::kNumAccum / 8 * 16
                    );
                }

                cute::tma_store_fence();

                // sync per-warpgroup rather than per-threadgroup and issue the TMA per warpgroup
                // this prevents both warpgroups from being idle at the same time as much as possible
                asm volatile("bar.sync %0, 128;\n" :: "r"(math_wg_idx));

                // Use TMA store to write back to global memory (per warpgroup rather than per threadgroup)
                // using threads which aren't part of a subprocessor that's active in the producer warpgroup
                if ((threadIdx.x == 96 || threadIdx.x == 224)) {
                    // TODO: evict_first is optimal when benchmarking the kernel individually
                    // but we might still want a way to disable it when the next kernel could realistically hit in the L2?
                    if (partially_oob) {
                        uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&tensor_map_d);
                        uint32_t smem_int_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_d + (math_wg_idx * BLOCK_N * 64)));
                        asm volatile (
                            "{\n"
                                ".reg .b64 policy;\n"
                                "createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                                "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%2, %3}], [%1], policy;\n"
                            "}\n"
                            :: "l"(gmem_int_desc), "r"(smem_int_ptr),
                            "r"(old_n_block_idx * BLOCK_N), "r"(old_global_idx + (math_wg_idx * 64)) : "memory");
                    } else {
                        uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&tensor_map_d_padded);
                        uint32_t smem_int_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_d + (math_wg_idx * BLOCK_N_STORE * 64)));
                        asm volatile (
                            "{\n"
                                ".reg .b64 policy;\n"
                                "createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\n"
                                "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%2, %3, %4}], [%1], policy;\n"
                            "}\n"
                            :: "l"(gmem_int_desc), "r"(smem_int_ptr),
                            "n"(0), "r"(old_n_block_idx), "r"(old_global_idx + (math_wg_idx * 64)) : "memory");
                    }

                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();
                }
                asm volatile("bar.sync %0, 128;\n" :: "r"(math_wg_idx));
            }
        };

        // Persistently schedule over blocks
        while (fetch_next_tile(m_block_idx, n_block_idx)) {
            // determine the B-scales address for this tile & barrier idx/parity for this tile
            int barrier_scales_b = (scheduler.current_iter & 1);
            int parity_scales_b = (scheduler.current_iter & 2) ? 1 : 0; // init=0 for consumer
            float* smem_scales_b = smem_scales_b_base + barrier_scales_b * kNumScalesB;
            // Decide the number of scales B to use (varies when N != 128 and non-uniform ScaleB)
            uint32_t num_former_iters = BLOCK_N / 8;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
            }
            // wait for the TMA load to fill them
            full_barrier_scales_b_base[barrier_scales_b].wait(parity_scales_b);

            // persistent across calls (write in wgmma_prepare_scales, read in promote_with_scales)
            //float scale_0_0[2], scale_1_0[2], scale_0_1[2], scale_1_1[2];
            float scale_a_0[2], scale_a_1[2], scale_b_0[2], scale_b_1[2];

            // ---------------------------------------------------------------------------------
            // Lambda to execute WGMMA on tensor cores and prepare scales for promotion
            // ---------------------------------------------------------------------------------
            auto wgmma_prepare_scales = [&](int idx, float* scales_b, bool accum_same_scale=false) {
                // Precompute descriptors to reduce instructions in inner loop
                uint64_t desc_a = desc_a_base + (s * (SMEM_AB_SIZE_PER_STAGE >> 4));
                uint64_t desc_b = desc_b_base + (s * (SMEM_AB_SIZE_PER_STAGE >> 4));
                float* accum = idx ? accum1 : accum0;

                // Wait TMA arrivals
                full_barriers_base[s].wait(parity);

                // Commit WGMMA instructions
                for (int i = 0; i < WGMMA::kNumAccum; ++i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                for (int k = 0; k < (BLOCK_K / WGMMA::K); k++) {
                    WGMMA::wgmma(desc_a, desc_b, accum, k || accum_same_scale);
                    desc_a += (WGMMA::K >> 4);
                    desc_b += (WGMMA::K >> 4);
                }
                warpgroup_commit_batch();
                for (int i = 0; i < WGMMA::kNumAccum; ++i)
                    warpgroup_fence_operand(accum[i]);

                // Read A & B scales (OK between warpgroup_arrive and empty_barrier_arrive with WGMMA double-buffering)
                if (!accum_same_scale) {
                    scale_a_0[idx] = ld_shared(smem_scales_a_base + s*BLOCK_M + r_0);
                    scale_a_1[idx] = ld_shared(smem_scales_a_base + s*BLOCK_M + r_1);
                    scale_b_0[idx] = ld_shared(scales_b);
                    if (!kMustUseUniformedScaleB) {
                        scale_b_1[idx] = ld_shared(scales_b + SHAPE_K_SCALES);
                    }
                }
            };

            // ---------------------------------------------------------------------------------
            // Lambda to promote with scaling factors on CUDA cores
            // ---------------------------------------------------------------------------------
            auto promote_with_scales = [&](int idx, bool add = true) {
                float* dst = final_accum;
                float* src = idx ? accum1 : accum0;

                // Calculate scaling factors used when promoting into final_accum later in promote_with_scales
                float scale_0_0 = scale_a_0[idx] * scale_b_0[idx];
                float scale_1_0 = scale_a_1[idx] * scale_b_0[idx];
                float scale_0_1, scale_1_1;
                if (!kMustUseUniformedScaleB) {
                    scale_0_1 = scale_a_0[idx] * scale_b_1[idx];
                    scale_1_1 = scale_a_1[idx] * scale_b_1[idx];
                }
                for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                    dst[i*4+0] = (add ? dst[i*4+0] : 0) + (predicate ? scale_0_0 : scale_0_1) * src[i*4+0];
                    dst[i*4+1] = (add ? dst[i*4+1] : 0) + (predicate ? scale_0_0 : scale_0_1) * src[i*4+1];
                    dst[i*4+2] = (add ? dst[i*4+2] : 0) + (predicate ? scale_1_0 : scale_1_1) * src[i*4+2];
                    dst[i*4+3] = (add ? dst[i*4+3] : 0) + (predicate ? scale_1_0 : scale_1_1) * src[i*4+3];
                }
            };

            constexpr int idx_0 = 0;
            constexpr int idx_1 = DP_SCALE_256 ? 0 : 1;
            constexpr int idx_2 = DP_SCALE_256 ? 1 : 0;
            constexpr int idx_3 = DP_SCALE_256 ? 1 : 1;

            if constexpr (DOUBLE_PUMP) {
                // Double Pumped (new-ish path)
                assert(SHAPE_K_SCALES % 2 == 0);

                wgmma_prepare_scales(idx_0, smem_scales_b + 0, false);
                next_stage();
                wgmma_prepare_scales(idx_1, smem_scales_b + 1, DP_SCALE_256);

                int tile_s = last_s;
                final_accum_to_bf16();

                warpgroup_wait<1>();
                next_stage();

                if (old_n_block_idx != -1) {
                    if constexpr (kNumMathThreads > 128) {
                        asm volatile("bar.sync 2, %0;\n" :: "n"(kNumMathThreads));
                        if (math_wg_idx == 0) {
                            store_tile(tile_s, 0, WGMMA::kNumAccum, true);
                            empty_barrier_arrive(&empty_barriers_base[tile_s]);
                        }
                    } else {
                        store_tile(tile_s, 0, WGMMA::kNumAccum, true);
                        empty_barrier_arrive(&empty_barriers_base[tile_s]);
                    }

                    if constexpr (!DP_SCALE_256) { promote_with_scales(0, false); }
                    wgmma_prepare_scales(idx_2, smem_scales_b + 2, false);

                    if (kNumMathThreads > 128 && math_wg_idx == 1) {
                        store_tile(tile_s, 0, WGMMA::kNumAccum, true);
                        empty_barrier_arrive(&empty_barriers_base[tile_s]);
                    }
                } else {
                    empty_barrier_arrive(&empty_barriers_base[tile_s]);
                    if constexpr (!DP_SCALE_256) { promote_with_scales(0, false); }
                    wgmma_prepare_scales(idx_2, smem_scales_b + 2, false);
                }

                warpgroup_wait<1>();
                empty_barrier_arrive(&empty_barriers_base[last_s]);
                next_stage();

                if constexpr (!DP_SCALE_256) { promote_with_scales(1);}
                wgmma_prepare_scales(idx_3, smem_scales_b + 3, DP_SCALE_256);

                if constexpr (DP_SCALE_256)  { promote_with_scales(0, false); }
                warpgroup_wait<1>();
                empty_barrier_arrive(&empty_barriers_base[last_s]);
                next_stage();
                if constexpr (!DP_SCALE_256) { promote_with_scales(0); }

                #pragma unroll 2
                for (int k_loop = 4; k_loop < SHAPE_K_SCALES; k_loop += 4) {
                    wgmma_prepare_scales(idx_0, smem_scales_b + k_loop + 0, false);
                    warpgroup_wait<1>();
                    empty_barrier_arrive(&empty_barriers_base[last_s]);

                    if constexpr (!DP_SCALE_256) { promote_with_scales(1); }

                    next_stage();
                    wgmma_prepare_scales(idx_1, smem_scales_b + k_loop + 1, DP_SCALE_256);

                    if constexpr (DP_SCALE_256) { promote_with_scales(1); }
                    warpgroup_wait<1>();
                    empty_barrier_arrive(&empty_barriers_base[last_s]);
                    next_stage();
                    if constexpr (!DP_SCALE_256) { promote_with_scales(0);}

                    wgmma_prepare_scales(idx_2, smem_scales_b + k_loop + 2, false);
                    warpgroup_wait<1>();
                    empty_barrier_arrive(&empty_barriers_base[last_s]);

                    if constexpr (!DP_SCALE_256) { promote_with_scales(1); }

                    next_stage();
                    wgmma_prepare_scales(idx_3, smem_scales_b + k_loop + 3, DP_SCALE_256);

                    if constexpr (DP_SCALE_256)  { promote_with_scales(0); }
                    warpgroup_wait<0>(); // sigh, damnit nvcc! TODO: 1 but hacked to 0 via SASS because we know better
                    empty_barrier_arrive(&empty_barriers_base[last_s]);
                    next_stage();
                    if constexpr (!DP_SCALE_256) { promote_with_scales(0); }
                }

                // TODO: tail when K is not a multiple of 4
                if constexpr (SHAPE_K_SCALES == 4) {
                    warpgroup_wait<0>();
                }
                empty_barrier_arrive(&empty_barriers_base[last_s]);
                promote_with_scales(1);

            } else {
                // Not Double Pumped (old-ish path)
                // WGMMA 0
                wgmma_prepare_scales(0, smem_scales_b + 0);

                if constexpr (SHAPE_K_SCALES > 1) {
                    // WGMMA 1
                    next_stage();
                    wgmma_prepare_scales(1, smem_scales_b + 1);

                    // Wait for WGMMA 0 (not the one we just issued) and let the producer know it can reuse its memory
                    warpgroup_wait<1>();
                    if constexpr (kNumMathThreads > 128) {
                        asm volatile("bar.sync 2, %0;\n" :: "n"(kNumMathThreads));
                    }
                    if (old_n_block_idx != -1) {
                        store_tile(last_s);
                    }
                    empty_barrier_arrive(&empty_barriers_base[last_s]);
                } else {
                    // Special case: single K_BLOCK so we don't need any other WGMMAs and we can just wait on WGMMA 0
                    warpgroup_wait<0>();
                    if constexpr (kNumMathThreads > 128) {
                        asm volatile("bar.sync 2, %0;\n" :: "n"(kNumMathThreads));
                    }
                    if (old_n_block_idx != -1) {
                        store_tile(s);
                    }
                    empty_barrier_arrive(&empty_barriers_base[s]);
                }

                // Promote without accumulation for WGMMA 0 (so we don't need to clear the registers)
                promote_with_scales(0, false);

                // KEY LOOP: This is where most of the WGMMAs usually happen (1 iteration per 2 K_BLOCK)
                #pragma unroll kNumUnroll
                for (int k_loop = 2; k_loop < SHAPE_K_SCALES-1; k_loop += 2) {
                    next_stage();
                    wgmma_prepare_scales(0, smem_scales_b + k_loop);

                    warpgroup_wait<1>(); // Wait for previous WGMMA (but not the one we just issued) and notify producer
                    empty_barrier_arrive(empty_barriers_base + last_s);
                    promote_with_scales(1);

                    next_stage();
                    wgmma_prepare_scales(1, smem_scales_b + k_loop + 1);

                    // Workaround to avoid NVCC/ptxas's warning "wgmma.mma_async instructions are serialized"
                    // If we don't wait for all WGMMAs at loop boundary, compiler screws things up (on 12.8)
                    // TODO: is there any way at all to avoid this when not fully unrolled?
                    if (k_loop/2 % kNumUnroll == 0) warpgroup_wait<0>();
                    else warpgroup_wait<1>();
                    empty_barrier_arrive(empty_barriers_base + last_s);
                    promote_with_scales(0);
                }
                // TODO: do we need a "dynamic tail" to avoid "instructions are serialized" with partial unroll?

                next_stage();
                if constexpr (SHAPE_K_SCALES % 2 == 1 && SHAPE_K_SCALES > 1) {
                    // Special case: K_BLOCK is not a multiple of 2 (e.g. K=384 with K_BLOCK=128)
                    // We need to do one last WGMMA to complete the tile
                    wgmma_prepare_scales(0, smem_scales_b + SHAPE_K_SCALES - 1);
                    warpgroup_wait<1>(); // implicit in warpgroup_wait<0> workaround above
                    empty_barrier_arrive(empty_barriers_base + last_s);
                    promote_with_scales(1);

                    next_stage();
                    warpgroup_wait<0>();
                    empty_barrier_arrive(empty_barriers_base + last_s);
                    promote_with_scales(0);
                } else {
                    // Usual case: we just need to promote the results of the last WGMMA
                    warpgroup_wait<0>(); // implicit in warpgroup_wait<0> workaround above
                    empty_barrier_arrive(empty_barriers_base + last_s);
                    promote_with_scales(1);
                }
            }

            if (lane_idx == 0) { // Not a cluster barrier so can't use empty_barrier_arrive
                empty_barrier_scales_b_base[barrier_scales_b].arrive();
            }

            // Write D for this tile while processing the next tile in the next loop iteration
            // Need to wait for space to store D (reusing memory from a stage of A/B) and store n/idx
            old_global_idx = scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx);
            old_n_block_idx = n_block_idx;
        }

        if (scheduler.current_iter > 0) {
            // Store the final tile to global memory
            store_tile(s);
            empty_barrier_arrive(&empty_barriers_base[s]);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages, uint32_t kNumUnroll,
          uint32_t kNumTMAMulticast, GemmType kGemmType, uint32_t l2_hash_bits, bool l2_optimization,
          uint32_t FORCED_M = 0>
class Gemm {
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    Gemm() = default;

    static void run(__nv_bfloat16* gmem_d, __nv_fp8_e4m3* gmem_b,
                    float* scales_b, int* grouped_layout,
                    uint32_t shape_m,
                    const CUtensorMap& tma_a_desc,
                    const CUtensorMap& tma_b_desc,
                    const CUtensorMap& tma_scales_a_desc,
                    const CUtensorMap& tma_d_desc,
                    const CUtensorMap& tma_d_padded_desc,
                    cudaStream_t stream, int num_sms, uint32_t smem_size,
                    unsigned char* gpu_side_index) {
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 256;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumUnroll, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      (kNumTMAMulticast > 1) ? true : false, kGemmType, l2_hash_bits, l2_optimization,
                                      FORCED_M>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // This will leak but considered OK since it's less memory than the code itself!
        // Kernel has a sacred duty to return this memory as zero so it can easily be reused
        static int* zeroed_scratch = nullptr;
        if (zeroed_scratch == nullptr) {
            cudaMalloc(&zeroed_scratch, 256 * sizeof(int));
            cudaMemset(zeroed_scratch, 0, 256 * sizeof(int));
        }

        static bool init_side_index = false;
        static param_side_index_t param_sideaware;
        if constexpr (l2_optimization) {
            if (!init_side_index) {
                cudaMemcpy(param_sideaware.sm_side_and_idx, gpu_side_index, MAX_SM * sizeof(unsigned char), cudaMemcpyDeviceToHost);
                init_side_index = true;
            }
        }
        assert(reinterpret_cast<uint64_t>(gmem_b) % 8192 == 0);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        if constexpr (kNumTMAMulticast <= 1) {
            cudaLaunchAttribute attr;
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim = {1, 1, 1};
            config.attrs = &attr;
            config.numAttrs = 1;
            config.gridDim = num_sms;
            auto status = cudaLaunchKernelEx(&config, kernel,
                                            gmem_d, gmem_b, scales_b, grouped_layout, zeroed_scratch, shape_m,
                                            tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc, tma_d_padded_desc,
                                            0, num_sms, param_sideaware);
            DG_HOST_ASSERT(status == cudaSuccess);
        } /*else if ((SHAPE_N % (BLOCK_N * 8)) == 0) {
            // TODO: Add back support for Hybrid Cluster Size with L2 side optimization
            // [...] see older commits [...]
        }*/ else {
            cudaLaunchAttribute attr;
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim = {1, 1, 1};
            config.attrs = &attr;
            config.numAttrs = 1;

            config.gridDim = num_sms;
            attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
            config.stream = stream;
            auto status = cudaLaunchKernelEx(&config, kernel,
                                            gmem_d, gmem_b, scales_b, grouped_layout, zeroed_scratch, shape_m,
                                            tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc, tma_d_padded_desc,
                                            0, num_sms, param_sideaware);
            DG_HOST_ASSERT(status == cudaSuccess);
        }
    }

    template <typename T>
    static CUtensorMap make_2d_tma_a_desc(T* global_address, uint32_t shape_m) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_K, BLOCK_M, BLOCK_K);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_b_desc(T* global_address) {
        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                SHAPE_K, SHAPE_N * (kGemmType != GemmType::Normal ? kNumGroups : 1), BLOCK_K, BLOCK_N);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_d_desc(T* global_address, uint32_t shape_m) {
        // TODO - I'm going to be slightly too honest and admit it's 2AM and I can't remember why I replaced BLOCK_M with 64
        // but otherwise it crashes, so...
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_N, min(/*BLOCK_M*/ 64, shape_m), BLOCK_N,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_3d_tma_d_desc(T* global_address, uint32_t shape_m) {
        return make_3d_tma_padded_desc(global_address,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_N, min(/*BLOCK_M*/ 64, shape_m), BLOCK_N,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_scales_a_desc(T* global_address, uint32_t shape_m) {
        // Make TMA aligned to 16 bytes
        constexpr uint32_t kAlignment = 16 / sizeof(T);
        shape_m = ceil_div(shape_m, kAlignment) * kAlignment;

        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                shape_m, ceil_div(SHAPE_K, BLOCK_K) * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), BLOCK_M, 1,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_desc(
            T* global_address, Layout layout,
            uint32_t gmem_rows, uint32_t gmem_cols,
            uint32_t smem_rows, uint32_t smem_cols,
            CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
        if (layout == Layout::RowMajor) {
            uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
            uint32_t smem_dim[2] = {smem_cols, smem_rows};
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_cols * sizeof(T), smem_dim, swizzle_type);
        } else {
            uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
            uint32_t smem_dim[2] = {smem_rows, smem_cols};
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type);
        }
    }

    template <typename T>
    static CUtensorMap make_3d_tma_padded_desc(
            T* global_address,
            uint32_t gmem_rows, uint32_t gmem_cols,
            uint32_t smem_rows, uint32_t smem_cols,
            CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
        uint32_t padding = (smem_cols == 64 || smem_cols == 96 || smem_cols == 128) ? PADDING_N : 0;
        uint64_t gmem_dim[3] = {smem_cols, gmem_cols/smem_cols, gmem_rows};
        uint32_t smem_dim[3] = {smem_cols+padding, 1, smem_rows};
        uint64_t stride_in_bytes[2] = {smem_cols * sizeof(T), gmem_cols * sizeof(T)};
        return make_3d_tma_copy_desc(global_address, gmem_dim, stride_in_bytes, smem_dim, swizzle_type);
    }
};

};  // namespace deep_gemm

#pragma clang diagnostic pop
