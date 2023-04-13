#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

using namespace pybind11::literals;
using namespace torch::jit;

namespace torch_ops::ops::nucleus_sampling {

void add_torch_module(torch::Library &m);

} // namespace torch_ops::ops::nucleus_sampling
