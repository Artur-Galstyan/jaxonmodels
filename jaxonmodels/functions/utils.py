from itertools import repeat

import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float


def make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def make_ntuple(x: Any, n: int) -> tuple[Any, ...]:
    if isinstance(x, collections.abc.Iterable):  # pyright: ignore
        return tuple(x)
    return tuple(repeat(x, n))


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def dtype_to_str(dtype) -> str:
    if hasattr(dtype, "__name__"):
        # For simple types like jnp.float32, float, etc.
        return dtype.__name__

    # For more complex types like JaxArray with specific dtype
    dtype_str = str(dtype)

    # Remove common patterns to clean up the string
    if "<class '" in dtype_str:
        # Extract the name from "<class 'jax.numpy.float32'>" -> "float32"
        dtype_str = dtype_str.split("'")[1].split(".")[-1]

    return dtype_str


def patch_merging_pad(x: Float[Array, "H W C"]) -> Array:
    H, W, C = x.shape
    h_pad = H % 2
    w_pad = W % 2
    x = jnp.pad(x, ((0, h_pad), (0, w_pad), (0, 0)))
    x0 = x[0::2, 0::2, :]
    x1 = x[1::2, 0::2, :]
    x2 = x[0::2, 1::2, :]
    x3 = x[1::2, 1::2, :]
    x = jnp.concatenate([x0, x1, x2, x3], axis=-1)
    return x
