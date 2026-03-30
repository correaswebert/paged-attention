#include <torch/extension.h>

// 1. Forward declare the host functions that launch your kernels.
void launch_prefill(/* your args here */);
void launch_decode(/* your args here */);

// 2. Define the single module entry point
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefill", &launch_prefill, "PagedAttention Prefill Kernel");
    m.def("decode", &launch_decode, "PagedAttention Decode Kernel");
    m.def("update", &launch_update, "PagedAttention Update Kernel");
}
