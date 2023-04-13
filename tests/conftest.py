import functools
import random

import numpy as np
import pytest
import torch
from _pytest.python import Function, Metafunc


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)


@functools.lru_cache()
def has_gpu() -> bool:
    return torch.cuda.is_available()


@functools.lru_cache()
def has_mps() -> bool:
    return torch.backends.mps.is_available()


def pytest_runtest_setup(item: Function) -> None:
    for mark in item.iter_markers():
        if mark.name == "has_gpu" and not has_gpu():
            pytest.skip("Skipping because this test requires a GPU and none is available")


def pytest_collection_modifyitems(items: list[Function]) -> None:
    items.sort(key=lambda x: x.get_closest_marker("slow") is not None, reverse=True)


def pytest_generate_tests(metafunc: Metafunc) -> None:
    if "device" in metafunc.fixturenames:
        torch_devices = [torch.device("cpu")]
        if has_gpu():
            torch_devices.append(torch.device("cuda"))
        if has_mps():
            torch_devices.append(torch.device("mps"))
        metafunc.parametrize("device", torch_devices)
