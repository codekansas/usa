#include "torch_ops.h"

namespace torch_ops {

PYBIND11_MODULE(torch_ops, m) {}

TORCH_LIBRARY(torch_ops, m) {
  torch_ops::ops::nucleus_sampling::add_torch_module(m);
}

} // namespace torch_ops
