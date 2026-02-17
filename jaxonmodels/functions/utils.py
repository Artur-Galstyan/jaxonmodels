import os
import pathlib
from itertools import repeat

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float, PRNGKeyArray


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
    if isinstance(x, collections.abc.Iterable):  # ty:ignore[unresolved-reference]
        return tuple(x)
    return tuple(repeat(x, n))


def default_floating_dtype():
    if jax.config.read("jax_enable_x64"):
        return jnp.float64
    else:
        return jnp.float32


def default_training_dtype():
    return jnp.bfloat16


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


def default_init(
    key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any, lim: float
) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_dtype = jnp.finfo(dtype).dtype
        rkey, ikey = jax.random.split(key, 2)
        real = jax.random.uniform(rkey, shape, real_dtype, minval=-lim, maxval=lim)
        imag = jax.random.uniform(ikey, shape, real_dtype, minval=-lim, maxval=lim)
        return real.astype(dtype) + 1j * imag.astype(dtype)
    else:
        return jax.random.uniform(key, shape, dtype, minval=-lim, maxval=lim)


def param_summary(model: eqx.Module, print_summary: bool = False):
    leaves_with_path = jax.tree_util.tree_leaves_with_path(
        model, is_leaf=eqx.is_array_like
    )
    total = 0
    rows = []
    for path, leaf in leaves_with_path:
        if not hasattr(leaf, "shape"):
            continue
        name = "".join(
            str(p.key)
            if hasattr(p, "key")
            else str(p.idx)
            if hasattr(p, "idx")
            else str(p)
            for p in path
        )
        n = 1
        for s in leaf.shape:
            n *= s
        rows.append((name, str(leaf.shape), n))
        total += n

    if print_summary:
        name_w = max(len(r[0]) for r in rows)
        shape_w = max(len(r[1]) for r in rows)
        param_w = max(len(f"{r[2]:,}") for r in rows)

        header = f"{'Layer':<{name_w}}  {'Shape':<{shape_w}}  {'Params':>{param_w}}"
        sep = "-" * len(header)

        print(sep)
        print(header)
        print(sep)
        for name, shape, n in rows:
            print(f"{name:<{name_w}}  {shape:<{shape_w}}  {n:>{param_w},}")
        print(sep)
        print(f"{'Total':<{name_w}}  {'':<{shape_w}}  {total:>{param_w},}")
        print(sep)
    return total


def get_cache_path(model: str):
    jaxonmodels_dir = os.path.expanduser(f"~/.jaxonmodels/models/{model}")
    os.makedirs(jaxonmodels_dir, exist_ok=True)
    return pathlib.Path(jaxonmodels_dir)
