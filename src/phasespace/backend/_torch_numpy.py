"""NumPy-style array namespace over PyTorch for the functions phasespace uses."""

__all__ = [
    "all",
    "asarray",
    "atleast_1d",
    "broadcast_to",
    "concatenate",
    "cos",
    "equal",
    "expand_dims",
    "float64",
    "int64",
    "log",
    "max",
    "ones",
    "ones_like",
    "prod",
    "reshape",
    "sin",
    "sort",
    "sqrt",
    "square",
    "stack",
    "sum",
    "transpose",
    "where",
    "zeros",
    "zeros_like",
]

import builtins
import functools

import torch
from torch import (
    atleast_1d,
    cos,
    float64,
    int64,
    log,
    ones,
    ones_like,
    reshape,
    sin,
    sqrt,
    square,
    where,
    zeros,
    zeros_like,
)


def asarray(obj, dtype=None):
    if isinstance(obj, (list, tuple)) and builtins.any(
        isinstance(item, torch.Tensor) for item in obj
    ):
        return torch.stack([asarray(item, dtype=dtype) for item in obj])
    return torch.as_tensor(obj, dtype=dtype)


def _as_tensors(arrays):
    return [asarray(array) for array in arrays]


def stack(arrays, axis=0):
    tensors = _as_tensors(arrays)
    dtype = functools.reduce(torch.promote_types, (tensor.dtype for tensor in tensors))
    return torch.stack([tensor.to(dtype) for tensor in tensors], dim=axis)


def concatenate(arrays, axis=0):
    return torch.cat(_as_tensors(arrays), dim=axis)


def broadcast_to(array, shape):
    shape = (shape,) if isinstance(shape, int) else tuple(int(s) for s in shape)
    return torch.broadcast_to(asarray(array), shape)


def expand_dims(array, axis):
    return torch.unsqueeze(asarray(array), axis)


def transpose(array):
    array = asarray(array)
    return array.permute(*reversed(range(array.ndim)))


def sort(array, axis=-1):
    return torch.sort(array, dim=axis).values


def equal(x, y):
    return torch.eq(asarray(x), asarray(y))


def all(array, axis=None):
    array = asarray(array)
    return torch.all(array) if axis is None else torch.all(array, dim=axis)


def max(array, axis=None):
    array = asarray(array)
    return torch.max(array) if axis is None else torch.amax(array, dim=axis)


def sum(array, axis=None, keepdims=False):
    array = asarray(array)
    if axis is None:
        return torch.sum(array)
    return torch.sum(array, dim=axis, keepdim=keepdims)


def prod(array, axis=None, keepdims=False):
    array = asarray(array)
    if axis is None:
        return torch.prod(array)
    return torch.prod(array, dim=axis, keepdim=keepdims)
