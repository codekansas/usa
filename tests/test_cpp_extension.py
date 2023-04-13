import pytest
import torch


@pytest.mark.slow()
def test_cpp_extension() -> None:
    import project.cpp.torch_ops as torch_ops

    input_tensor = torch.randn(4, 16)
    output_tensor = torch_ops.nucleus_sampling(input_tensor, 0.1)
    assert output_tensor.shape == (4, 1)
