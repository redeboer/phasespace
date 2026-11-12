__all__ = [
    "Generator",
    "default_rng",
    "from_seed",
    "uniform",
]

from numbers import Integral

import torch
from torch import Generator


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
