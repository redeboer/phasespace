"""PyTorch implementation of the backend interface.

PyTorch runs eagerly on the CPU in double precision. Its array namespace lives in `._torch_numpy`.
"""

from numbers import Integral

import torch
from torch import Generator

from . import _torch_numpy
from ._numpy import assert_equal, assert_greater_equal
from ._types import Implementation, Random


def create_implementation() -> Implementation:
    return Implementation(
        array=_torch_numpy,
        random=Random(
            Generator=Generator,
            default_rng=default_rng,
            from_seed=from_seed,
            uniform=uniform,
        ),
        Tensor=torch.Tensor,
        Variable=torch.Tensor,
        shape=shape,
        assert_equal=assert_equal,
        assert_greater_equal=assert_greater_equal,
    )


def shape(array):
    return tuple(array.shape)


def from_seed(seed) -> Generator:
    generator = Generator()
    generator.manual_seed(int(seed))
    return generator


def default_rng() -> Generator:
    return torch.default_generator


def uniform(rng, shape, minval, maxval, dtype):
    shape = (shape,) if isinstance(shape, Integral) else tuple(int(s) for s in shape)
    sample = torch.rand(shape, generator=rng, dtype=dtype)
    return minval + (maxval - minval) * sample
