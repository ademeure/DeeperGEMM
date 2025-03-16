// TODO - rewrite all of this!!!
// need to change how L2 SM/page side info is tested and passed around
// (could be sideband or hidden inside parity bits, but careful page side implicitly depends on SM side!)
// This should be fused with FP8 conversion/transpose kernels if possible
// and done persistently so it's written from a SM on the correct side

constexpr uint32_t l2_hash_bits = 0x0018AB000; // for GH200 96GiB (TODO: upper bits are redundant)
//constexpr uint32_t l2_hash_bits = 0x0018B3000; // for H100 80GiB

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
template <uint32_t BLOCK_K, GemmType kGemmType>
__global__ void optimize_B(__nv_fp8_e4m3* gmem_b_out, __nv_fp8_e4m3* gmem_b_in, int* grouped_layout,
                           const __grid_constant__ CUtensorMap tensor_map_b,
                           uint32_t* gpu_l2_sides, uint8_t* gpu_page_l2_sides,
                           int shape_n, int shape_k, int num_groups) {
    // Currently don't support L2 side optimization for grouped masked GEMM (possible in theory?)
    constexpr bool L2_SIDE_OPTIMIZATION = (kGemmType != GemmType::GroupedMasked);
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
        //printf("group_idx: %d, offset: %d\n", group_idx, offset);

        int start_page_offset = reinterpret_cast<uint64_t>(gmem_b_out) % (2048*1024);
        int page_idx = (offset + start_page_offset) / (2048*1024);
        int page_side = gpu_page_l2_sides[page_idx];
        int local_side =  __popc(reinterpret_cast<uint64_t>(&gmem_b_out[offset]) & l2_hash_bits) & 1;
        int upper_4kib = n_side ^ page_side ^ local_side;

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            //printf("N: %d(/%d), N_HALF: %d, N_SIDE: %d, N_DST_BASE: %d, K: %d(/%d), OFFSET: %d, PAGE_IDX: %d, PAGE_SIDE: %d, LOCAL_SIDE: %d, UPPER_4KIB: %d\n", n, shape_n, n_half, n_side, n_dst_base, k, shape_k, offset, page_idx, page_side, local_side, upper_4kib);
        }

        int4* address = reinterpret_cast<int4*>(gmem_b_out + (offset + upper_4kib * 4096));
        address[threadIdx.x] = data_int4;
    } else {
        // simple memcpy for non-optimized case
        int offset = tid * 16;
        uint4* out4 = reinterpret_cast<uint4*>(gmem_b_out + offset);
        *out4 = *(reinterpret_cast<uint4*>(gmem_b_in + offset));
    }
}

constexpr int L2_SIDE_TEST_ITERATIONS = 100;
template<int MAX_SM=132>
__global__ void l2_side_per_sm(uint32_t* sm_side, uint32_t* data) {
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
        int offset = smid;
        uint32_t old_value = atomicExch(&data[offset], 0); // backup old value

        long long int start_clock = clock64();
        for (int i = 0; i < L2_SIDE_TEST_ITERATIONS; i++) {
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
                sm_side[i+MAX_SM+1] = sm_side[i] / L2_SIDE_TEST_ITERATIONS;
                sm_side[i] = (sm_side[i] > average_latency) ? 1 : 0;
            }
            sm_side[MAX_SM] = average_latency;
        }
        data[offset] = old_value; // restore old value
    }
}

template<int MAX_SM=132>
__global__ void l2_side_per_page(uint8_t* l2_side, uint32_t* data, const uint32_t* sm_side, int num_pages) {
    __nanosleep(1000);
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid) :);
        // get side of this SM
        int side = sm_side[smid];
        int average_latency = sm_side[MAX_SM];

        int start_offset = reinterpret_cast<uint64_t>(data) % (2048*1024);
        uint32_t* data_page_start = data - (start_offset / sizeof(uint32_t));

        for (int p = smid; p < num_pages; p += gridDim.x) {
            uint32_t offset = p * (2048*1024/sizeof(uint32_t)); // 2MiB pages
            if (p == 0) offset = start_offset / sizeof(uint32_t);
            uint32_t old_value = atomicExch(&data_page_start[offset], 0); // backup old value

            long long int start_clock = clock64();
            if (start_clock == 0) start_clock = clock64();
            for (int i = 0; i < L2_SIDE_TEST_ITERATIONS; i++) {
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
            l2_side[p + num_pages] = total_latency / (10*L2_SIDE_TEST_ITERATIONS); // for debugging
        }
    }
}

template <uint32_t BLOCK_K, GemmType kGemmType>
class ReorderB {
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    ReorderB() = default;

    static void run(__nv_fp8_e4m3* gmem_b_out, __nv_fp8_e4m3* gmem_b_in, int* grouped_layout, const CUtensorMap& tma_b_desc_in,
                    uint32_t shape_n, uint32_t shape_k, cudaStream_t stream, int num_sms, int num_groups) {

        uint32_t kNumTiles = ceil_div(shape_n, 8U) * ceil_div(shape_k, BLOCK_K) * num_groups;
        //printf("!!!!! kNumTiles: %d (n: %d, k: %d, groups: %d)\n", kNumTiles, shape_n, shape_k, num_groups);

        // not currently used, no benefit when combined with larger/hybrid cluster sizes? :(
        constexpr int MAX_SM = 132;
        static __nv_fp8_e4m3* previous_b = nullptr;
        static uint32_t* gpu_l2_sides = nullptr;
        if (gpu_l2_sides == nullptr || previous_b != gmem_b_out) {
            previous_b = gmem_b_out;
            if (gpu_l2_sides != nullptr) cudaFree(gpu_l2_sides);
            cudaMalloc(&gpu_l2_sides, (MAX_SM*2+1) * sizeof(uint32_t));
            cudaMemset(gpu_l2_sides, 0, (MAX_SM*2+1) * sizeof(uint32_t));
            l2_side_per_sm<MAX_SM><<<num_sms, 128>>>(gpu_l2_sides, (uint32_t*)gmem_b_out);

            uint32_t cpu_l2_sides[MAX_SM*2+1];
            cudaMemcpy(cpu_l2_sides, gpu_l2_sides, (MAX_SM*2+1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            int side_1 = 0;
            for (int i = 0; i < MAX_SM; i++) {
                side_1 += cpu_l2_sides[i];
                //printf("SM %d: %d (%d)\n", i, cpu_l2_sides[i], cpu_l2_sides[i+MAX_SM+1]);
            }
            //printf("SM0 Side: %d\n", cpu_l2_sides[0]);
            //printf("SM Side 0+1: %d+%d\n", num_sms - side_1, side_1);
            //printf("Average latency: %d\n", cpu_l2_sides[MAX_SM] / L2_SIDE_TEST_ITERATIONS);
        }

        // Is the address 8KiB aligned?
        uint64_t address = reinterpret_cast<uint64_t>(gmem_b_out);
        // printf("Address 2MiB alignment: %d (n: %d, k: %d, num_groups: %d)\n", (int)(address % (2048*1024)), shape_n, shape_k, num_groups);
        assert(address % 8192 == 0);

        int num_pages = ceil_div(shape_n * shape_k * num_groups + (address % (2048*1024)), 2048UL*1024UL);
        uint8_t* gpu_page_l2_sides = nullptr;
        cudaMalloc(&gpu_page_l2_sides, 2*num_pages * sizeof(uint8_t));
        cudaMemset(gpu_page_l2_sides, 0, 2*num_pages * sizeof(uint8_t));
        l2_side_per_page<<<128, MAX_SM>>>(gpu_page_l2_sides, (uint32_t*)gmem_b_out, gpu_l2_sides, num_pages);

        /*uint8_t* cpu_page_l2_sides = new uint8_t[2*num_pages];
        cudaMemcpy(cpu_page_l2_sides, gpu_page_l2_sides, 2*num_pages * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        printf("\n");
        for (int i = 0; i < num_pages; i++) {
            printf("%2d ", cpu_page_l2_sides[i]);
        }
        printf("\n\n");
        free(cpu_page_l2_sides);*/

        // Calculate number of tiles and smem size
        constexpr uint32_t smem_size = BLOCK_K * 8 * sizeof(__nv_fp8_e4m3) + 1024;

        auto kernel = optimize_B<BLOCK_K, kGemmType>;
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

        auto status = cudaLaunchKernelEx(&config, kernel, gmem_b_out, gmem_b_in, grouped_layout, tma_b_desc_in, gpu_l2_sides, gpu_page_l2_sides, shape_n, shape_k, num_groups);
        DG_HOST_ASSERT(status == cudaSuccess);

        cudaFree(gpu_page_l2_sides);
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
