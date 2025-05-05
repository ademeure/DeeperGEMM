// TODO - rewrite all of this!!!
// need to change how L2 SM/page side info is tested and passed around
// (could be sideband or hidden inside parity bits, but careful page side implicitly depends on SM side!)
// This should be fused with FP8 conversion/transpose kernels if possible
// and done persistently so it's written from a SM on the correct side


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm {

enum class Layout {
    RowMajor,
    ColMajor
};
enum class GemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};

// TODO: L2 side aware B with 128x32 blocks (contiguous 4KiB)
template <uint32_t BLOCK_K, GemmType kGemmType, uint32_t l2_hash_bits, bool l2_optimization>
__global__ void optimize_B(__nv_fp8_e4m3* gmem_b_out, __nv_fp8_e4m3* gmem_b_in, int* grouped_layout,
                           const __grid_constant__ CUtensorMap tensor_map_b,
                           int shape_n, int shape_k, int num_groups) {
    // Currently don't support L2 side optimization for grouped masked GEMM (possible in theory?)
    constexpr bool L2_SIDE_OPTIMIZATION = l2_optimization && (kGemmType != GemmType::GroupedMasked);
    uint32_t shape_n_half = shape_n / 2; // this works because N%64==0 and fp8_gemm.cuh loads in chunks of 32xN

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    int laneid;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid));

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    Barrier* barrier = reinterpret_cast<Barrier*>(smem_buffer + 1024);

    if constexpr (L2_SIDE_OPTIMIZATION) {
        int blocks_per_group = (shape_n/8U) * (shape_k/BLOCK_K);
        int group_idx = blockIdx.x / blocks_per_group;
        int idx_in_group = blockIdx.x % blocks_per_group;
        int n = (idx_in_group * 8) % shape_n;
        int k = ((idx_in_group * 8) / shape_n) * 128;

        if (threadIdx.x == 0 && laneid == 0) {
            barrier->init(1);
            cutlass::arch::fence_view_async_shared();
            tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(barrier), smem_buffer, k, n + (group_idx * shape_n));
            barrier->arrive_and_expect_tx(1024);
        }
        __syncthreads();
        barrier->wait(0);

        int4 data_int4 = reinterpret_cast<int4*>(smem_buffer)[threadIdx.x];
        __nv_fp8_e4m3 *data_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(&data_int4);

        int n_side = (n >= shape_n_half) ? 1 : 0;
        int n_half = n_side ? (n - shape_n_half) : n;
        int n_dst_base = (n_half & 31) + (n_half & ~31) * 2;
        int offset = (n_dst_base * 128) + (k * shape_n);

        offset += group_idx * shape_n * shape_k;

        int local_side =  __popc(reinterpret_cast<uint64_t>(&gmem_b_out[offset]) & l2_hash_bits) & 1;
        int upper_4kib = n_side ^ local_side ^ 1; // extra ^1 with new sideaware.cu because bit 21 is in the hash

        int4* address = reinterpret_cast<int4*>(gmem_b_out + (offset + upper_4kib * 4096));
        address[threadIdx.x] = data_int4;
    } else {
        // simple memcpy for non-optimized case
        int offset = tid * 16;
        uint4* out4 = reinterpret_cast<uint4*>(gmem_b_out + offset);
        *out4 = *(reinterpret_cast<uint4*>(gmem_b_in + offset));
    }
}

template <uint32_t BLOCK_K, GemmType kGemmType, uint32_t l2_hash_bits, bool l2_optimization>
class ReorderB {
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    ReorderB() = default;

    static void run(__nv_fp8_e4m3* gmem_b_out, __nv_fp8_e4m3* gmem_b_in, int* grouped_layout, const CUtensorMap& tma_b_desc_in,
                    uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_sms, int num_groups) {
        // Is the address 8KiB aligned?
        uint64_t address = reinterpret_cast<uint64_t>(gmem_b_out);
        assert(address % 8192 == 0);

        // Calculate number of tiles and smem size
        uint32_t kNumTiles = ceil_div(shape_n, 8U) * ceil_div(shape_k, BLOCK_K) * num_groups;
        constexpr uint32_t smem_size = BLOCK_K * 8 * sizeof(__nv_fp8_e4m3) + 1024;

        auto kernel = optimize_B<BLOCK_K, kGemmType, l2_hash_bits, l2_optimization>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Launch
        cudaLaunchConfig_t config;
        config.blockDim = 64;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
        config.gridDim = kNumTiles;

        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {1, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;

        auto status = cudaLaunchKernelEx(&config, kernel, gmem_b_out, gmem_b_in, grouped_layout, tma_b_desc_in, shape_n, shape_k, num_groups);
        DG_HOST_ASSERT(status == cudaSuccess);

        cudaDeviceSynchronize();
    }

    template <typename T>
    static CUtensorMap make_2d_tma_b_desc(T* global_address, int shape_n, int shape_k, int num_groups=1) {
        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                shape_k, shape_n * (kGemmType != GemmType::Normal ? num_groups : 1), BLOCK_K, 8);
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
