#include <torch/extension.h>

// Forward declarations
torch::Tensor custom_flash_attention(torch::Tensor q, torch::Tensor k,
                                     torch::Tensor v, int num_heads,
                                     bool causal);
torch::Tensor custom_flash_attention_decode(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, int block_size, torch::Tensor context_lens,
    int num_heads, bool causal);
void update_kv_cache(torch::Tensor k_cache, torch::Tensor v_cache,
                     torch::Tensor k, torch::Tensor v,
                     torch::Tensor block_table, int block_size,
                     torch::Tensor current_positions);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefill", &custom_flash_attention, "PagedAttention Prefill Kernel");
    m.def("decode", &custom_flash_attention_decode,
          "PagedAttention Decode Kernel");
    m.def("update", &update_kv_cache, "PagedAttention Update Kernel");
}
