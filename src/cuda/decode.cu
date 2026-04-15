#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <torch/extension.h>

using namespace torch::indexing;

template <int B>
__global__ void flash_attention_decode_kernel(
    const half* __restrict__ q, const half* __restrict__ k_cache,
    const half* __restrict__ v_cache, const int32_t* __restrict__ block_table,
    half* __restrict__ out, const int32_t* __restrict__ context_lens,
    int head_dim, int num_heads, int block_size, int max_num_blocks,
    int max_context_len, bool causal) {
    __shared__ float q_tile[128];
    __shared__ float k_tile[B][128];
    __shared__ float v_tile[B][128];
    __shared__ float o_tile[128];
    __shared__ float m_tile;
    __shared__ float l_tile;
    __shared__ float scores[B];
    __shared__ float p[B];

    int bh_idx = blockIdx.x;
    int tid = threadIdx.x;

    int b_idx = bh_idx / num_heads;
    int h_idx = bh_idx % num_heads;

    int context_len = context_lens[b_idx];

    const half* q_bh = q + bh_idx * head_dim;
    half* o_bh = out + bh_idx * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int i = tid; i < head_dim; i += blockDim.x) {
        q_tile[i] = __half2float(q_bh[i]);
        o_tile[i] = 0.0f;
    }
    if (tid == 0) {
        m_tile = -FLT_MAX;
        l_tile = 0.0f;
    }
    __syncthreads();

    int num_kv_blocks = (context_len + B - 1) / B;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        int kv_start = kv_block_idx * B;

        if (tid < B) {
            int kv_row = kv_start + tid;
            if (kv_row < context_len) {
                int logical_block_idx = kv_row / block_size;
                int block_offset = kv_row % block_size;
                int physical_block_idx =
                    block_table[b_idx * max_num_blocks + logical_block_idx];

                long k_offset = (long)physical_block_idx *
                                    (block_size * num_heads * head_dim) +
                                (long)block_offset * (num_heads * head_dim) +
                                (long)h_idx * head_dim;

                for (int d = 0; d < head_dim; d++) {
                    k_tile[tid][d] = __half2float(k_cache[k_offset + d]);
                    v_tile[tid][d] = __half2float(v_cache[k_offset + d]);
                }
            } else {
                for (int d = 0; d < head_dim; d++) {
                    k_tile[tid][d] = 0.0f;
                    v_tile[tid][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (tid < B) {
            int kv_col = kv_start + tid;
            float score = 0.0f;
            if (kv_col < context_len) {
                for (int d = 0; d < head_dim; d++) {
                    score += q_tile[d] * k_tile[tid][d];
                }
                score *= scale;
            } else {
                score = -FLT_MAX;
            }
            scores[tid] = score;
        }
        __syncthreads();

        if (tid == 0) {
            float m_curr = -FLT_MAX;
            for (int j = 0; j < B; j++) {
                if (scores[j] > m_curr) {
                    m_curr = scores[j];
                }
            }
            float m_new = fmaxf(m_tile, m_curr);
            float l_row = 0.0f;
            for (int j = 0; j < B; j++) {
                float pval = expf(scores[j] - m_new);
                p[j] = pval;
                l_row += pval;
            }
            float alpha = expf(m_tile - m_new);
            float l_new = alpha * l_tile + l_row;
            for (int d = 0; d < head_dim; d++) {
                float o_val = alpha * o_tile[d];
                for (int j = 0; j < B; j++) {
                    o_val += p[j] * v_tile[j][d];
                }
                o_tile[d] = o_val;
            }
            m_tile = m_new;
            l_tile = l_new;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float l_inv = 1.0f / l_tile;
        for (int d = 0; d < head_dim; d++) {
            o_bh[d] = __float2half(o_tile[d] * l_inv);
        }
    }
}

torch::Tensor custom_flash_attention_decode(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, int block_size, torch::Tensor context_lens,
    int num_heads, bool causal) {
    auto options = q.options();

    int batch_size = q.size(0);
    int hidden_dim = q.size(2);
    int head_dim = hidden_dim / num_heads;
    int max_num_blocks = block_table.size(1);

    auto q_bh = q.view({batch_size, 1, num_heads, head_dim})
                    .permute({0, 2, 1, 3})
                    .contiguous()
                    .view({batch_size * num_heads, 1, head_dim});

    auto out_bh = torch::zeros_like(q_bh, options);

    int max_context_len = context_lens.max().item<int>();

    const int B = 32;
    int total_batch_heads = batch_size * num_heads;

    dim3 grid(total_batch_heads);
    dim3 block(B);

    flash_attention_decode_kernel<32>
        <<<grid, block>>>(reinterpret_cast<half*>(q_bh.data_ptr<at::Half>()),
                          reinterpret_cast<half*>(k_cache.data_ptr<at::Half>()),
                          reinterpret_cast<half*>(v_cache.data_ptr<at::Half>()),
                          block_table.data_ptr<int32_t>(),
                          reinterpret_cast<half*>(out_bh.data_ptr<at::Half>()),
                          context_lens.data_ptr<int32_t>(), head_dim, num_heads,
                          block_size, max_num_blocks, max_context_len, causal);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    auto out = out_bh.view({batch_size, num_heads, 1, head_dim})
                   .permute({0, 2, 1, 3})
                   .contiguous()
                   .view({batch_size, 1, hidden_dim});

    return out;
}

__global__ void update_cache_kernel(
    half* __restrict__ k_cache, half* __restrict__ v_cache,
    const half* __restrict__ k, const half* __restrict__ v,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ current_positions, int batch_size, int head_dim,
    int num_heads, int block_size, int max_num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hidden_dim = head_dim * num_heads;
    int total_elements = batch_size * hidden_dim;
    if (idx >= total_elements) return;

    int b = idx / hidden_dim;
    int d = idx % hidden_dim;
    int h_idx = d / head_dim;
    int d_idx = d % head_dim;

    int src_idx = b * hidden_dim + d;

    // The sequence position indicating exactly where to write this newly
    // produced KV:
    int current_pos = current_positions[b];

    int logical_block_idx = current_pos / block_size;
    int block_offset = current_pos % block_size;
    int physical_block_idx =
        block_table[b * max_num_blocks + logical_block_idx];

    long dst_idx =
        (long)physical_block_idx * (block_size * num_heads * head_dim) +
        (long)block_offset * (num_heads * head_dim) + (long)h_idx * head_dim +
        (long)d_idx;

    k_cache[dst_idx] = k[src_idx];
    v_cache[dst_idx] = v[src_idx];
}

void update_kv_cache(torch::Tensor k_cache, torch::Tensor v_cache,
                     torch::Tensor k, torch::Tensor v,
                     torch::Tensor block_table, int block_size,
                     torch::Tensor current_positions) {
    int batch_size = k.size(0);
    int num_heads = k_cache.size(2);
    int head_dim = k_cache.size(3);
    int max_num_blocks = block_table.size(1);

    int total_threads = batch_size * num_heads * head_dim;
    int threads_per_block = 256;
    dim3 threads(threads_per_block);
    dim3 blocks((total_threads + threads_per_block - 1) / threads_per_block);

    update_cache_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v.data_ptr<at::Half>()),
        block_table.data_ptr<int32_t>(), current_positions.data_ptr<int32_t>(),
        batch_size, head_dim, num_heads, block_size, max_num_blocks);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
