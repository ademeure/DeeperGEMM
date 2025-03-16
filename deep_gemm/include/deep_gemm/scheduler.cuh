#include "utils.cuh"

namespace deep_gemm {

enum class GemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups,
          // TODO: halved because of L2 side awareness effectively doubling Ns in parallel
          // but this should probably be auto-tuned instead...
          uint32_t kNumNBlocksPerGroup = 8,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N)>
struct Scheduler {
    int current_iter = -1;
    uint32_t num_aligned_m_blocks;

    // For normal GEMM
    // Maybe not used in the masked grouped GEMM
    uint32_t num_blocks;

    // For grouped GEMM
    int* grouped_layout;
    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum;
    // with hybrid cluster sizes, we can't use blockIdx.x/gridDim.x directly
    // e.g. with 15 clusters of 8 + 4 clusters of 2, latter will have: block_idx = 120+blockIdx.x
    int block_idx, grid_size;
    int n_block_offset;
    __device__ __forceinline__ explicit Scheduler(const uint32_t shape_m,
                                                  int* grouped_layout = nullptr,
                                                  int block_idx = -1, int grid_size = -1,
                                                  int n_block_offset = 0) {
        num_aligned_m_blocks = ceil_div(shape_m, BLOCK_M);
        if constexpr (kGemmType == GemmType::Normal) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
            this->grouped_layout = grouped_layout;
        } else if (kGemmType == GemmType::GroupedMasked) {
            curr_group_idx = curr_cumsum = 0;
            this->grouped_layout = grouped_layout;
        }
        this->block_idx = block_idx >= 0 ? block_idx : blockIdx.x;
        this->grid_size = grid_size >= 0 ? grid_size : gridDim.x;
        this->n_block_offset = n_block_offset;
    }

    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t num_m_blocks, int block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        // TODO: check for this statically host side (since cluster size is now dynamic)
        //DG_STATIC_ASSERT(kNumNBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        auto num_blocks_per_group = num_m_blocks * kNumNBlocksPerGroup; // HACK: TODO: temporary optimization for m=64
        auto group_idx = block_idx / num_blocks_per_group;
        auto first_n_block_idx = group_idx * kNumNBlocksPerGroup;
        auto num_n_blocks_in_group = min(kNumNBlocksPerGroup, kNumNBlocks - first_n_block_idx);
        auto in_group_idx = block_idx % num_blocks_per_group;
        m_block_idx = in_group_idx / num_n_blocks_in_group;
        n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
    }

    template <bool kIgnoreGroupedForGroupedContiguous=true>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx=0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            auto offset = kIgnoreGroupedForGroupedContiguous ? 0 : __ldg(grouped_layout + m_block_idx * BLOCK_M);
            return offset * shape_dim + block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedMasked) {
            return curr_group_idx * shape_dim + block_idx * block_size;
        }
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * grid_size + block_idx;

        if constexpr (kGemmType == GemmType::GroupedMasked) {
            uint32_t num_m_blocks;
            while (true) {
                // End of the task
                if (curr_group_idx == kNumGroups) {
                    m_block_idx = 0xFFFFFFFF;
                    return false;
                }

                // Within current group
                num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(grouped_layout + curr_group_idx)), BLOCK_M);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                    break;

                // Move to check the next group
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
        } else {
            if (next_block_idx >= num_blocks) {
                m_block_idx = 0xFFFFFFFF;
                return false;
            }

            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
        }
        n_block_idx += n_block_offset;
        return true;
    }
};
#pragma clang diagnostic pop

} // namespace deep_gemm
