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

enum class Layout {
    RowMajor,
    ColMajor
};

template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return (block_m == 64 ? 1 : 2) * kNumMathThreadsPerGroup + kNumTMAThreads;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages, uint32_t kNumUnroll,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          bool     kTMAMulticastEnabled,
          GemmType kGemmType>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                uint32_t shape_m,
                const __grid_constant__ CUtensorMap tensor_map_a,
                const __grid_constant__ CUtensorMap tensor_map_b,
                const __grid_constant__ CUtensorMap tensor_map_scales_a,
                const __grid_constant__ CUtensorMap tensor_map_d,
                int block_idx_base, int aggregate_grid_size) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_N, BLOCK_K) == 1, "Too much B scales in a single block");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Shared memory
    static constexpr int kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_AB_SIZE_PER_STAGE = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
    static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_B_SIZE = ceil_div<uint32_t>(SHAPE_K_SCALES * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads) {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

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
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_b_base) + SMEM_SCALES_B_SIZE);
    auto full_barriers_base = barrier_start_ptr;
    auto empty_barriers_base = barrier_start_ptr + kNumStages;
    auto full_barrier_scales_b_base = barrier_start_ptr + kNumStages * 2;
    auto empty_barrier_scales_b_base = barrier_start_ptr + kNumStages * 2 + 2; // double-buffered

    uint32_t cluster_size;
    // get cluster_nctaid
    asm volatile("mov.u32 %0, %cluster_nctaid.x;" : "=r"(cluster_size));

    // Initialize barriers
    if (threadIdx.x == kNumMathThreads) {
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            full_barriers_base[i].init(1);
            empty_barriers_base[i].init(cluster_size * kNumMathThreads / 32);
        }
        for (int i = 0; i < 2; ++ i) {
            full_barrier_scales_b_base[i].init(1);
            empty_barrier_scales_b_base[i].init(kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kTMAMulticastEnabled) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kTMAMulticastEnabled) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    constexpr int kNumTMARegisters = 24;
    constexpr int kNumMathRegisters = 240;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups>
                              (shape_m, grouped_layout, block_idx_base + blockIdx.x, aggregate_grid_size);

    // Updates pipeline stage & parity in as few SASS instructions as possible
    int s = 0, last_s = -1, parity = -1; // persistent context across loop iterations
    auto next_stage = [&]() {
        bool wrap = (s == kNumStages-1);
        last_s = s;
        s = wrap ? 0 : s + 1;
        parity = wrap ? !parity : parity;
    };

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        parity = 1; // producer starts with parity=1 (no wait)
        elect_or_exit(); // tell nvcc this is single-threaded (bad codegen otherwise on 12.8)

        // NOTES: thread 0 for loading A/B/A_scales per-K_BLOCK, thread 32 to load B_scales per-tile
        // these are 2 different subprocessors, and we use threads on another subprocessor for TMA stores later
        if (threadIdx.x == kNumMathThreads) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                if (cute::block_rank_in_cluster() == 0 || !kTMAMulticastEnabled) {
                    #pragma unroll 1
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
                        // Issue TMA B without broadcasting
                        tma_copy(&tensor_map_b, full_barrier64, smem_b_base + s * SMEM_AB_SIZE_PER_STAGE,
                                  k_idx * BLOCK_K, scheduler.get_global_idx<false>(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx), cluster_size);
                        full_barrier.arrive_and_expect_tx(SMEM_AB_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                        next_stage();
                    }
                } else {
                    // Other parts of the cluster don't need to load A (since it has been multicast by rank 0)
                    // We specialise the code so that it's fully efficient even without loop unrolling
                    #pragma unroll 1
                    for (int k_idx = 0; k_idx < SHAPE_K_SCALES; k_idx++) {                        
                        // Wait consumer release
                        empty_barriers_base[s].wait(parity);
                        // Issue TMA B without broadcasting
                        tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barriers_base[s]), smem_b_base + s * SMEM_AB_SIZE_PER_STAGE,
                                  k_idx * BLOCK_K, scheduler.get_global_idx<false>(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx));
                        full_barriers_base[s].arrive_and_expect_tx(SMEM_AB_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
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
        } else if (threadIdx.x == kNumMathThreads + 32) {
            // Load scales B via TMA Load per-tile rather than per-K_BLOCK
            // previously done with global memory loads, which was OK but forced synchronization between warpgroups
            // hardcoded to always be doubled-buffered (equivalent to s=2) which is more than enough (but s=1 isn't!)
            #pragma unroll 1
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
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
        auto store_tile = [&] (int old_n_block_idx, uint32_t old_global_idx, int tile_s) {
            // Write final_accum to shared memory using STSM
            auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_a_base + last_s * (SMEM_AB_SIZE_PER_STAGE));

            // Write back to shared memory using STSM
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 8; ++ i) {
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[i * 8 + 0], final_accum[i * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 2], final_accum[i * 8 + 3]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 4], final_accum[i * 8 + 5]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 6], final_accum[i * 8 + 7]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16)
                );
            }
            if constexpr (WGMMA::kNumAccum % 8 != 0) {
                SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 0], final_accum[WGMMA::kNumAccum / 8 * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 2], final_accum[WGMMA::kNumAccum / 8 * 8 + 3]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + WGMMA::kNumAccum / 8 * 16
                );
            }
            cute::tma_store_fence();

            // sync per-warpgroup rather than per-threadgroup and issue the TMA per warpgroup
            // this prevents both warpgroups from being idle at the same time as much as possible
            asm volatile("bar.sync %0, 128;\n" :: "r"(math_wg_idx));

            // Use TMA store to write back to global memory (per warpgroup rather than per threadgroup)
            // using threads which aren't part of a subprocessor that's active in the producer warpgroup
            if (threadIdx.x == 96 || threadIdx.x == 224) {
                // TODO: evict_first is optimal when benchmarking the kernel individually
                // but we might still want a way to disable it when the next kernel could realistically hit in the L2?
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
                
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            __syncwarp();

            // notify the producer we no longer need this memory for D and it can be reused for A/B
            empty_barrier_arrive(&empty_barriers_base[tile_s]);
        };

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {            
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
            float scale_0_0[2], scale_1_0[2], scale_0_1[2], scale_1_1[2];

            // ---------------------------------------------------------------------------------
            // Lambda to execute WGMMA on tensor cores and prepare scales for promotion
            // ---------------------------------------------------------------------------------
            auto wgmma_prepare_scales = [&](int idx, float* scales_b) {
                // Precompute descriptors to reduce instructions in inner loop
                uint64_t desc_a = desc_a_base + ((s * SMEM_AB_SIZE_PER_STAGE) >> 4);
                uint64_t desc_b = desc_b_base + ((s * SMEM_AB_SIZE_PER_STAGE) >> 4);
                float* accum = idx ? accum1 : accum0;

                // Wait TMA arrivals
                full_barriers_base[s].wait(parity);

                // Commit WGMMA instructions
                for (int i = 0; i < WGMMA::kNumAccum; ++i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                for (int k = 0; k < (BLOCK_K / WGMMA::K); k++) {
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                    desc_a += (WGMMA::K >> 4);
                    desc_b += (WGMMA::K >> 4);
                }
                warpgroup_commit_batch();
                for (int i = 0; i < WGMMA::kNumAccum; ++i) warpgroup_fence_operand(accum[i]);

                // Read A & B scales (OK between warpgroup_arrive and empty_barrier_arrive with WGMMA double-buffering)
                float scale_a_0 = ld_shared(smem_scales_a_base + s*BLOCK_M + r_0);
                float scale_a_1 = ld_shared(smem_scales_a_base + s*BLOCK_M + r_1);
                float scale_b_0 = ld_shared(scales_b);
                // Calculate scaling factors used when promoting into final_accum later in promote_with_scales
                scale_0_0[idx] = scale_a_0 * scale_b_0;
                scale_1_0[idx] = scale_a_1 * scale_b_0;
                if (!kMustUseUniformedScaleB) {
                    float scale_b_1 = ld_shared(scales_b + SHAPE_K_SCALES);
                    scale_0_1[idx] = scale_a_0 * scale_b_1;
                    scale_1_1[idx] = scale_a_1 * scale_b_1;
                }
            };

            // ---------------------------------------------------------------------------------
            // Lambda to promote with scaling factors on CUDA cores
            // ---------------------------------------------------------------------------------
            auto promote_with_scales = [&](int idx, bool add = true) {
                float* dst = final_accum;
                float* src = idx ? accum1 : accum0;
                for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                    dst[i*4+0] = (add ? dst[i*4+0] : 0) + (predicate ? scale_0_0[idx] : scale_0_1[idx]) * src[i*4+0];
                    dst[i*4+1] = (add ? dst[i*4+1] : 0) + (predicate ? scale_0_0[idx] : scale_0_1[idx]) * src[i*4+1];
                    dst[i*4+2] = (add ? dst[i*4+2] : 0) + (predicate ? scale_1_0[idx] : scale_1_1[idx]) * src[i*4+2];
                    dst[i*4+3] = (add ? dst[i*4+3] : 0) + (predicate ? scale_1_0[idx] : scale_1_1[idx]) * src[i*4+3];
                }
            };

            // WGMMA 0
            wgmma_prepare_scales(0, smem_scales_b + 0);

            // Overlap writes of the previous MxN tile with the processing of WGMMA 0 of the current tile
            // (This function is also called at the end for the very last tile of the workload)
            // last_s => storage space for D (& s => input data for WGMMA in flight above)
            if (old_n_block_idx != -1) {
                store_tile(old_n_block_idx, old_global_idx, last_s);
            }

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

            // Key loop where most of the WGMMAs usually happen (1 iteration per 2 K_BLOCK)
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
                //warpgroup_wait<0>(); // implicit in warpgroup_wait<0> workaround above
                empty_barrier_arrive(empty_barriers_base + last_s);
                promote_with_scales(1);
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
            store_tile(old_n_block_idx, old_global_idx, last_s);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}


// TODO: L2 side aware B with 128x32 blocks (contiguous 4KiB)
template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages, GemmType kGemmType>
__global__ void optimize_B(int* grouped_layout,
                           const __grid_constant__ CUtensorMap tensor_map_b_in,
                           const __grid_constant__ CUtensorMap tensor_map_b_out) {
}


template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages, uint32_t kNumUnroll,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType>
class Gemm {
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    Gemm() = default;

    static void run(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                    uint32_t shape_m,
                    const CUtensorMap& tma_a_desc,
                    const CUtensorMap& tma_b_desc,
                    const CUtensorMap& tma_scales_a_desc,
                    const CUtensorMap& tma_d_desc,
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumUnroll, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      (kNumTMAMulticast > 1) ? true : false, kGemmType>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

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
                                            gmem_d, scales_b, grouped_layout,
                                            shape_m,
                                            tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
                                            0, num_sms);
            DG_HOST_ASSERT(status == cudaSuccess);
        } else if (num_sms >= 132 && (SHAPE_N % (BLOCK_N * 8)) == 0) {
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

            if (ceil_div(SHAPE_N, (uint32_t)128)*100 <= ceil_div(SHAPE_N, (uint32_t)132)*105) {
                // use 128 SMs instead of 132 for better cache locality
                // when it doesn't noticeably increase the number of "waves" (outer loop iterations)
                // TODO: commenting this out triggered an assert at one point, so there might be a bug elsewhere
                num_sms = 128;
            }

            static cudaStream_t stream1 = nullptr, stream2 = nullptr;
            static cudaEvent_t event1 = nullptr, event2 = nullptr;
            if (stream1 == nullptr) {
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
                                             gmem_d, scales_b, grouped_layout,
                                             shape_m,
                                             tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
                                             0, num_sms);

            // use the remaining SMs with a cluster size of 2 threadgroups
            config.gridDim = num_sms - (clusters * 8);
            attr.val.clusterDim = {2, 1, 1};
            config.stream = stream2;
            auto status2 = cudaLaunchKernelEx(&config, kernel,
                                            gmem_d, scales_b, grouped_layout,
                                            shape_m,
                                            tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
                                            clusters * 8, num_sms);
            DG_HOST_ASSERT(status1 == cudaSuccess && status2 == cudaSuccess);
            
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
                                            gmem_d, scales_b, grouped_layout,
                                            shape_m,
                                            tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
                                            0, num_sms);
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
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_N, min(64 /*BLOCK_M*/, shape_m), BLOCK_N,
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
};

};  // namespace deep_gemm

#pragma clang diagnostic pop
