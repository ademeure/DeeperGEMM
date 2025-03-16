


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
constexpr int NUM_WARPS_LOADING_B = 2; // number of threads reading B in parallel, 2 max right now (4 didn't help)
constexpr int L2_TEST_ITERATIONS = 100; // todo - rewrite the L2/page testing code and pass as sideband?
constexpr int PADDING_N = 16; // padding for D to avoid STSM bank conflicts (todo - clearer conditions etc.)
constexpr bool DOUBLE_PUMP = true; // todo - figure out how we can make this *always* faster (not just usually so...)
constexpr bool DP_SCALE_256 = false; // todo - assumes A/B scales are always the same for 2 blocks, need test data here
constexpr int NUM_TILES_INITIAL = 32; // calxulate m/n for INITIAL tiles in parallel in prologue
constexpr int NUM_TILES_STORAGE = 64; // 1 more every time we load B scales in a round-robin buffer

constexpr uint32_t l2_hash_bits = 0x0018AB000; // for GH200 96GiB (TODO: upper bits are redundant)
//constexpr uint32_t l2_hash_bits = 0x0018B3000; // for H100 80GiB

enum class Layout {
    RowMajor,
    ColMajor
};

template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return (block_m == 64 ? 1 : 2) * kNumMathThreadsPerGroup + kNumTMAThreads;
}

typedef struct { // todo - redo this whole thing properly
    uint8_t sm_side_and_idx[133];
    uint8_t page_l2_sides[1024];
} param_large_t;

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages, uint32_t kNumUnroll,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          bool     kTMAMulticastEnabled, uint32_t kNumBLoaders,
          GemmType kGemmType,
          uint32_t NUM_PAGES, uint32_t MAX_SM, uint32_t FORCED_M>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel(__nv_bfloat16* gmem_d, __nv_fp8_e4m3* gmem_b, float* scales_b, int* grouped_layout, int* zeroed_scratch,
                uint32_t shape_m,
                const __grid_constant__ CUtensorMap tensor_map_a,
                const __grid_constant__ CUtensorMap tensor_map_b,
                const __grid_constant__ CUtensorMap tensor_map_scales_a,
                const __grid_constant__ CUtensorMap tensor_map_d,
                const __grid_constant__ CUtensorMap tensor_map_d_padded,
                int block_idx_base, int aggregate_grid_size,
                __grid_constant__ const param_large_t large_params) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    //
    constexpr bool L2_SIDE_OPTIMIZATION = (kGemmType != GemmType::GroupedMasked);
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
        int side = large_params.sm_side_and_idx[smid] & 1;
        scheduler.block_idx = large_params.sm_side_and_idx[smid] >> 1;
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
        m_block_idx = smem_tile_scheduling[scheduler.current_iter].x;
        n_block_idx = smem_tile_scheduling[scheduler.current_iter].y;
        if constexpr (kGemmType == GemmType::GroupedMasked) {
            scheduler.curr_group_idx = smem_tile_scheduling[scheduler.current_iter].z;
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
            parity ^= 1;
        }
    };


    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-groups for loading data - we split the threads into
        // 1) Calculating future tile m/n and loading B scales per tile (1 thread)
        // 2) Loading A data & scales (1 thread)
        // 3) Loading B data (multiple threads/warps, expensive due to L2 side awareness, needs redesign)
        // Unlike the others, (3) supports using multiple warps & multiple threads per warp to parallelise the L2 side calculations
        // Trying >2 warps for (3) by increasing TMA threads to 256 didn't help (extra threads weren't the issue)
        // Really need to rearchitect (3) to use TMA tensor loads instead of TMA 1D loads (and possibly add alignment constraints)
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        parity = 1; // producer starts with parity=1 (no wait)

        // NOTES: thread 0 for loading A/B/A_scales per-K_BLOCK, thread 32 to load B_scales per-tile
        // these are 2 different subprocessors, and we use threads on another subprocessor for TMA stores later
        if (warp_idx >= kNumMathWarps && warp_idx < kNumMathWarps+kNumBLoaders) {
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
            int loader_idx = kNumBLoaders > 1 ? warp_idx-kNumMathWarps : 0;
            int loader_tid = threadIdx.x-kNumMathThreads;

            auto load_b_data = [&](int start_k_idx) {
                if (kNumBLoaders > 1 && start_k_idx > 0) next_stage();
                if (kNumBLoaders > 2 && start_k_idx > 1) next_stage();
                if (kNumBLoaders > 3 && start_k_idx > 2) next_stage();

                if constexpr (L2_SIDE_OPTIMIZATION) {
                    int current_shape_n = (scheduler.n_block_offset) ? SHAPE_N : SHAPE_N_LOWER;
                    int start_page_offset = reinterpret_cast<uint64_t>(gmem_b) % (2048*1024);
                    __nv_fp8_e4m3* b_page_start = gmem_b - start_page_offset;
                    smem_b_base += (NUM_CHUNKS > 1) ? lane_idx * (CHUNK_SIZE * BLOCK_K) : 0;

                    int global_base_offset = start_page_offset + (start_k_idx * BLOCK_K * SHAPE_N);
                    int lane_chunk_start = (NUM_CHUNKS > 1) ? (lane_idx * CHUNK_SIZE) : 0;

                    // Persistently schedule over blocks to load B
                    #pragma unroll 1
                    while (fetch_next_tile(m_block_idx, n_block_idx)) {
                        int n = n_block_idx * BLOCK_N;
                        int remaining_n = current_shape_n - n;
                        int n_to_load_warp = max(0, min(remaining_n, BLOCK_N));
                        int n_to_load_lane = (n_to_load_warp - lane_chunk_start);

                        int num_bytes_total = n_to_load_warp * BLOCK_K;
                        int num_bytes = (n_to_load_lane > CHUNK_SIZE) ? (CHUNK_SIZE*BLOCK_K) : (n_to_load_lane*BLOCK_K);

                        n += lane_chunk_start;
                        int n_side = (n >= SHAPE_N_HALF) ? 1 : 0;
                        int n_half = (n_side * (-SHAPE_N_HALF)) + n;
                        int n_dst_base = n_half + (n_half & ~31); // shift everything after bit 5 to the left by 1
                        int tile_base_offset = global_base_offset + (n_dst_base * 128);
                        if constexpr (kGemmType == GemmType::GroupedContiguous) {
                            int group_offset = __ldg(grouped_layout + m_block_idx * BLOCK_M);
                            tile_base_offset += (SHAPE_N * SHAPE_K) * group_offset;
                        }

                        #pragma unroll 1
                        for (int k_idx = start_k_idx; k_idx < SHAPE_K_SCALES; k_idx += kNumBLoaders, tile_base_offset += BLOCK_K * SHAPE_N * kNumBLoaders) {
                            // Wait consumer release
                            auto& full_barrier = full_barriers_base[s];
                            uint64_t* full_barrier64 = reinterpret_cast<uint64_t*>(&full_barrier);
                            empty_barriers_base[s].wait(parity);

                            if (num_bytes > 0) {
                                __nv_fp8_e4m3* smem_address = smem_b_base + s * SMEM_AB_SIZE_PER_STAGE;
                                int offset = tile_base_offset;
                                int page_idx = offset >> 21;
                                uint32_t page_side = large_params.page_l2_sides[page_idx];
                                uint32_t address_lower_bits = (reinterpret_cast<uint64_t>(b_page_start) & 0xFFFFFFFF) + offset;
                                int local_side =  __popc((address_lower_bits & l2_hash_bits) ^ n_side) & 1;
                                int upper_4kib = local_side ^ page_side;
                                __nv_fp8_e4m3* address = b_page_start + (offset + upper_4kib * 4096);
                                // TMA instructions are for a single thread so compiler will automatically generate a loop here
                                // This is potentially still faster than looping over the entire thing
                                // since we can calculate page/address/etc. in parallel this way
                                cute::SM90_BULK_COPY_G2S::copy(address, full_barrier64, smem_address, num_bytes);
                            }
                            if (lane_idx == 0) {
                                full_barriers_base[s].arrive_and_expect_tx(num_bytes_total);
                            }

                            // Increment by multiple stages at once (only works with the assert below)
                            DG_STATIC_ASSERT(kNumStages > kNumBLoaders, "kNumStages must be greater than kNumBLoaders");
                            s = s + kNumBLoaders;
                            if (s >= kNumStages) {
                                s -= kNumStages;
                                parity ^= 1;
                            }
                        }

                        if (loader_tid == 0) {
                            // Used for D (which now reuses an A/B pipeline stage instead of dedicated memory)
                            empty_barriers_base[s].wait(parity);
                            full_barriers_base[s].arrive();
                        }
                        next_stage();
                    }
                } else {
                    // legacy approach without L2 side optimization
                    while (fetch_next_tile(m_block_idx, n_block_idx)) {
                        for (int k_idx = start_k_idx; k_idx < SHAPE_K_SCALES; k_idx += kNumBLoaders) {
                            auto& full_barrier = full_barriers_base[s];
                            uint64_t* full_barrier64 = reinterpret_cast<uint64_t*>(&full_barrier);
                            empty_barriers_base[s].wait(parity);
                            tma_copy(&tensor_map_b, full_barrier64, smem_b_base + s * SMEM_AB_SIZE_PER_STAGE,
                                    k_idx * BLOCK_K, scheduler.get_global_idx<false>(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx), cluster_size);
                            full_barriers_base[s].arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                            s = s + kNumBLoaders;
                            if (s >= kNumStages) {
                                s -= kNumStages;
                                parity ^= 1;
                            }
                        }
                        if (loader_tid == 0) {
                            empty_barriers_base[s].wait(parity);
                            full_barriers_base[s].arrive();
                        }
                        next_stage();
                    }

                }
            };

            // Call the lambda with the appropriate starting k_idx
            load_b_data(loader_idx);

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kTMAMulticastEnabled) {
                if (loader_idx == 0 && lane_idx == 0) {
                    for (int i = 0; i < kNumStages; i++) {
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
                // Used for D (which now reuses an A/B pipeline stage instead of dedicated memory)
                empty_barriers_base[s].wait(parity);
                full_barriers_base[s].arrive();
                next_stage();
            }
            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kTMAMulticastEnabled) {
                for (int i = 0; i < kNumStages; i++) {
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
                //////
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
        auto store_tile = [&] (int tile_s, int start=0, int end=WGMMA::kNumAccum) {
            int current_shape_n = (scheduler.n_block_offset) ? SHAPE_N : SHAPE_N_LOWER;

            // Write final_accum to shared memory using STSM
            // Padded to avoid up to 8x(!) shared memory bank conflicts
            auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_a_base + tile_s * (SMEM_AB_SIZE_PER_STAGE));
            bool partially_oob = (old_n_block_idx * BLOCK_N) > (SHAPE_N - BLOCK_N);
            uint32_t BLOCK_N_STORE = partially_oob ? BLOCK_N : BLOCK_N_PADDED;

            // Only process part of the tile at a time if possible
            if (start == 0) final_accum_to_bf16();

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
                __syncwarp();

                // notify the producer we no longer need this memory for D and it can be reused for A/B
                empty_barrier_arrive(&empty_barriers_base[tile_s]);
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
                assert(SHAPE_K_SCALES % 2 == 0);
                auto tile_s = last_s;
                int quarter_tile = WGMMA::kNumAccum / 4;

                wgmma_prepare_scales(idx_0, smem_scales_b + 0, false);
                 if (old_n_block_idx != -1) store_tile(tile_s);

                next_stage();
                wgmma_prepare_scales(idx_1, smem_scales_b + 1, DP_SCALE_256);

                warpgroup_wait<1>();
                empty_barrier_arrive(&empty_barriers_base[last_s]);
                next_stage();

                if constexpr (!DP_SCALE_256) { promote_with_scales(0, false); }

                wgmma_prepare_scales(idx_2, smem_scales_b + 2, false);

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
                // WGMMA 0
                wgmma_prepare_scales(0, smem_scales_b + 0);

                // Overlap writes of the previous MxN tile with the processing of WGMMA 0 of the current tile
                // (This function is also called at the end for the very last tile of the workload)
                // last_s => storage space for D (& s => input data for WGMMA in flight above)
                if (old_n_block_idx != -1) {
                    store_tile(last_s);
                }// else if (old_n_block_idx != -1) {
                //    empty_barrier_arrive(&empty_barriers_base[last_s]);
                //}

                if constexpr (SHAPE_K_SCALES > 1) {
                    // WGMMA 1
                    next_stage();
                    wgmma_prepare_scales(1, smem_scales_b + 1);

                    // Wait for WGMMA 0 (not the one we just issued) and let the producer know it can reuse its memory
                    warpgroup_wait<1>();
                    empty_barrier_arrive(&empty_barriers_base[last_s]);
                } else {
                    // Special case: single K_BLOCK so we don't need any other WGMMAs and we can just wait on WGMMA 0
                    warpgroup_wait<0>();
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
            full_barriers_base[s].wait(parity);
            next_stage();
        }

        if (scheduler.current_iter > 0) {
            // Store the final tile to global memory

            store_tile(last_s);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

template<int MAX_SM=132>
__global__ void l2_side_per_sm(uint32_t* sm_side, uint32_t* data) {
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
        int offset = smid;
        uint32_t old_value = atomicExch(&data[offset], 0); // backup old value

        long long int start_clock = clock64();
        for (int i = 0; i < L2_TEST_ITERATIONS; i++) {
            int value = atomicAdd(&data[offset], 1);
            offset += (value > MAX_SM*1000) ? 1 : 0; // impossible condition to make the compiler play along
        }
        int total_latency = clock64() - start_clock;
        sm_side[smid] = total_latency;
        atomicAdd(&sm_side[MAX_SM], total_latency);

        int num_done = atomicAdd(&sm_side[MAX_SM+1], 1);
        if (num_done == gridDim.x - 1) {
            int average_latency = sm_side[MAX_SM] / gridDim.x;
            for (int i = 0; i < MAX_SM; i++) {
                sm_side[i] = (sm_side[i] > average_latency) ? 1 : 0;
            }
            sm_side[MAX_SM] = average_latency;
        }
        data[offset] = old_value; // restore old value
    }
}

template<int MAX_SM=132>
__global__ void l2_side_per_page(uint32_t* l2_side, uint32_t* data, const uint32_t* sm_side, int num_pages, int num_bytes) {
    __nanosleep(1000);
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);

        // get side of this SM
        int side = sm_side[smid];
        int average_latency = sm_side[MAX_SM];

        int start_offset = (reinterpret_cast<uint64_t>(data) % (2048*1024)) / sizeof(uint32_t);
        uint32_t* data_page_start = data - start_offset;

        for (int p = smid; p < num_pages; p += gridDim.x) {
            uint32_t offset = p * (2048*1024/sizeof(uint32_t)); // 2MiB pages
            if (p == 0) {
                offset = start_offset;
            }
            else if (offset - start_offset >= num_bytes / sizeof(uint32_t)) {
                return;
            }

            uint32_t old_value = atomicExch(&data_page_start[offset], 0); // backup old value

            long long int start_clock = clock64();
            if (start_clock == 0) start_clock = clock64();
            for (int i = 0; i < L2_TEST_ITERATIONS; i++) {
                int value = atomicAdd(&data_page_start[offset], 1);
                offset += (value > 100000) ? 1 : 0; // impossible condition to make the compiler play along
            }
            int total_latency = clock64() - start_clock;
            if (total_latency == 0) total_latency = clock64() - start_clock;
            data_page_start[offset] = old_value; // restore old value

            bool near_side = total_latency <= average_latency;
            int bitmask_side =  (__popcll((reinterpret_cast<uint64_t>(&data_page_start[offset])) & l2_hash_bits) & 1);
            int final_side = side ^ near_side ^ bitmask_side;

            l2_side[p] = final_side;
            l2_side[p + num_pages] = total_latency / (10*L2_TEST_ITERATIONS); // for debugging
            //printf("start_offset: %d, offset: %d, average_latency: %d, total_latency: %d ==> %d\n", start_offset, offset, average_latency, total_latency, l2_side[p]);
        }
    }
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages, uint32_t kNumUnroll,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType,
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
                    cudaStream_t stream, int num_sms, uint32_t smem_size) {
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 256;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        constexpr int MAX_SM = 132;

        constexpr int num_pages = (SHAPE_N * SHAPE_K * kNumGroups + 2048U*1024U - 1U) / (2048U*1024U) + 1;
        assert(reinterpret_cast<uint64_t>(gmem_b) % 8192 == 0);
        assert(num_pages <= 1024);

        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumUnroll, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      (kNumTMAMulticast > 1) ? true : false, NUM_WARPS_LOADING_B, kGemmType, num_pages, MAX_SM, FORCED_M>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // this *will* leak but considered OK for now since it's less memory than the code itself!
        // kernel has a sacred duty to return this memory as zero so it can easily be reused
        static int* zeroed_scratch = nullptr;
        if (zeroed_scratch == nullptr) {
            cudaMalloc(&zeroed_scratch, 256 * sizeof(int));
            cudaMemset(zeroed_scratch, 0, 256 * sizeof(int));
        }

        static __nv_fp8_e4m3* previous_b = nullptr;
        static uint32_t* gpu_l2_sides = nullptr;
        static uint32_t cpu_l2_sides[MAX_SM+1];
        static uint32_t sm_idx_on_side[MAX_SM];
        static uint32_t cpu_page_l2_sides[num_pages];

        if (gpu_l2_sides == nullptr || previous_b != gmem_b) {
            previous_b = gmem_b;
            if (gpu_l2_sides != nullptr) cudaFree(gpu_l2_sides);
            cudaMalloc(&gpu_l2_sides, (MAX_SM+2) * sizeof(uint32_t));
            cudaMemset(gpu_l2_sides, 0, (MAX_SM+2) * sizeof(uint32_t));
            l2_side_per_sm<MAX_SM><<<MAX_SM, 128>>>(gpu_l2_sides, (uint32_t*)gmem_b);

            cudaMemcpy(cpu_l2_sides, gpu_l2_sides, (MAX_SM+1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            int side_0 = 0, side_1 = 0;
            for (int i = 0; i < MAX_SM; i++) {
                int side = cpu_l2_sides[i];
                sm_idx_on_side[i] = side ? side_1 : side_0;
                side_0 += !side;
                side_1 += side;
                //printf("(FP8 GEMM) SM %d: %d\n", i, side);
            }
            //printf("(FP8 GEMM) SM0 Side: %d\n", cpu_l2_sides[0]);
            //printf("(FP8 GEMM) SM Side 0+1: %d+%d\n", num_sms - side_0, side_1);

            uint32_t* gpu_page_l2_sides ;
            cudaMalloc(&gpu_page_l2_sides, 2*num_pages * sizeof(uint32_t));
            cudaMemset(gpu_page_l2_sides, 0, 2*num_pages * sizeof(uint32_t));

            l2_side_per_page<<<128, 128>>>(gpu_page_l2_sides, (uint32_t*)gmem_b, gpu_l2_sides, num_pages, SHAPE_N * SHAPE_K * kNumGroups);
            cudaMemcpy(cpu_page_l2_sides, gpu_page_l2_sides, num_pages * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaFree(gpu_page_l2_sides);

            //printf("\n");
            for (int i = 0; i < num_pages; i++) {
                //printf("%2d ", cpu_page_l2_sides[i]);
            }
            //printf("\n\n");
        }

        param_large_t large_params;
        for (int i = 0; i < MAX_SM; i++) large_params.sm_side_and_idx[i] = sm_idx_on_side[i] << 1 | cpu_l2_sides[i];
        for (int i = 0; i < num_pages; i++) large_params.page_l2_sides[i] = cpu_page_l2_sides[i];
        large_params.sm_side_and_idx[MAX_SM] = cpu_l2_sides[MAX_SM] / L2_TEST_ITERATIONS;

        // Cluster launch
        cudaLaunchConfig_t config;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // TODO: Explain "Hybrid Cluster Size" properly
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
                                            0, num_sms, large_params);
            DG_HOST_ASSERT(status == cudaSuccess);
        } else if (num_sms >= 999 && (SHAPE_N % (BLOCK_N * 8)) == 0) {
            // TODO: Fix with L2 side optimization
            // requires using atomicAdd to get index dynamically (+add idx inside cluster)
            // and using base_idx to start at a certain number for the 2nd set

            // use 128 SMs instead of 132 for better cache locality
            // and because otherwise everything is a mess...
            num_sms = 128;

            // use as many clusters of 8 threadgroups as possible as they are lower power
            // TODO: if num_sms < SMs and another kernel is using the reserved SMs, we might have a problem:
            // if the other kernel is scheduled first, AFAIK there's no guarantee the *right SMs* are still available!
            cudaLaunchAttribute attr;
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
            config.attrs = &attr;
            config.numAttrs = 1;

            int clusters = 0;
            config.gridDim = 8;
            attr.val.clusterDim = {8, 1, 1};
            cudaOccupancyMaxActiveClusters(&clusters, kernel, &config);

            static cudaStream_t stream1 = nullptr, stream2 = nullptr;
            static cudaEvent_t event1 = nullptr, event2 = nullptr;
            if (stream1 == nullptr) {
                // this will leak, assuming it's OK/negligible for now
                // (CUDA has no real limit but reuses HW resources which may lead to false dependencies)
                cudaStreamCreate(&stream1);
                cudaStreamCreate(&stream2);
                cudaEventCreate(&event1);
                cudaEventCreate(&event2);
            }

            // make stream1 and stream2 dependent on stream
            cudaEventRecord(event1, stream);
            cudaStreamWaitEvent(stream1, event1, 0);
            cudaStreamWaitEvent(stream2, event1, 0);

            config.gridDim = clusters * 8;
            config.stream = stream1;
            auto status1 = cudaLaunchKernelEx(&config, kernel,
                                             gmem_d, gmem_b, scales_b, grouped_layout, zeroed_scratch, shape_m,
                                             tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc, tma_d_padded_desc,
                                             0, num_sms, large_params);
            DG_HOST_ASSERT( status1 == cudaSuccess);

            // use the remaining SMs with a cluster size of 2 threadgroups
            config.gridDim = num_sms - (clusters * 8);
            attr.val.clusterDim = {2, 1, 1};
            config.stream = stream2;
            auto status2 = cudaLaunchKernelEx(&config, kernel,
                                            gmem_d, gmem_b, scales_b, grouped_layout, zeroed_scratch, shape_m,
                                            tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc, tma_d_padded_desc,
                                            clusters * 8, num_sms, large_params);
            DG_HOST_ASSERT( status2 == cudaSuccess);

            // make stream dependent on stream1 and stream2
            cudaEventRecord(event1, stream1);
            cudaEventRecord(event2, stream2);
            cudaStreamWaitEvent(stream, event1, 0);
            cudaStreamWaitEvent(stream, event2, 0);
        } else {
            cudaLaunchAttribute attr;
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim = {1, 1, 1};
            config.attrs = &attr;
            config.numAttrs = 1;

            // TODO: need to add support for hybrid cluster sizes with reserved SMs but it's tricky
            // if num_sms < SMs and another kernel is using the reserved SMs, we might have a problem:
            // if the other kernel is scheduled first, AFAIK there's no guarantee the *right SM* are still available!
            // ==> one solution might be to issue extra clusters of 2 and dynamically get a block idx using atomicAdd
            // so if the large clusters haven't launched after nanosleep(ns), they start execution, otherwise they exit?
            // but also(/rather) provide a way to make sure the other kernels use SMs we don't need to avoid this issue
            config.gridDim = num_sms;
            attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
            config.stream = stream;
            auto status = cudaLaunchKernelEx(&config, kernel,
                                            gmem_d, gmem_b, scales_b, grouped_layout, zeroed_scratch, shape_m,
                                            tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc, tma_d_padded_desc,
                                            0, num_sms, large_params);
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
