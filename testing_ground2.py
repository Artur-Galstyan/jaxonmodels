import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from beartype.typing import Sequence
from jaxtyping import Array, PRNGKeyArray

from jaxonmodels.functions.utils import default_floating_dtype

np.random.seed(42)


x = np.array(np.random.normal(size=(56, 56, 96)))

jax_layer_norm = eqx.nn.LayerNorm(96)
torch_layer_norm = torch.nn.LayerNorm(96)


t_out = torch_layer_norm(torch.Tensor(x)).detach().numpy()
print(t_out.shape)


j_out1 = eqx.filter_vmap(eqx.filter_vmap(jax_layer_norm))(jnp.array(x))
print(j_out1.shape)

print(np.allclose(t_out, np.array(j_out1), atol=1e-5))


class LayerNorm(eqx.Module):
    normalized_shape: tuple[int, ...] = eqx.field(static=True)
    axes: tuple[int, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    use_weight: bool = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    weight: Array | None
    bias: Array | None

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        use_weight: bool = True,
        use_bias: bool = True,
        dtype=None,
    ):
        if isinstance(normalized_shape, int):
            shape_tuple = (normalized_shape,)
        else:
            shape_tuple = tuple(normalized_shape)
        self.normalized_shape = shape_tuple
        self.axes = tuple(range(-len(self.normalized_shape), 0))
        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias

        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None

        if self.use_weight:
            self.weight = jnp.ones(self.normalized_shape, dtype=dtype)
        else:
            self.weight = None
        if self.use_bias:
            self.bias = jnp.zeros(self.normalized_shape, dtype=dtype)
        else:
            self.bias = None

    def __call__(
        self,
        x: Array,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        if x.shape[-len(self.normalized_shape) :] != self.normalized_shape:
            raise ValueError(
                f"Input shape {x.shape} must end "
                "with normalized_shape {self.normalized_shape}"
            )

        orig_dtype = x.dtype
        with jax.numpy_dtype_promotion("standard"):
            calc_dtype_elems = [x.dtype, jnp.float32]
            if self.weight is not None:
                calc_dtype_elems.append(self.weight.dtype)
            if self.bias is not None:
                calc_dtype_elems.append(self.bias.dtype)
            calc_dtype = jnp.result_type(*calc_dtype_elems)

        x = x.astype(calc_dtype)

        mean = jnp.mean(x, axis=self.axes, keepdims=True)
        mean_keepdims = jnp.mean(x, axis=self.axes, keepdims=True)
        variance = jnp.mean(
            jnp.square(x - mean_keepdims), axis=self.axes, keepdims=True
        )
        variance = jnp.maximum(0.0, variance)
        inv = jax.lax.rsqrt(variance + self.eps)
        out = (x - mean) * inv

        if self.use_weight:
            assert self.weight is not None
            out = out * self.weight.astype(calc_dtype)
        if self.use_bias:
            assert self.bias is not None
            out = out + self.bias.astype(calc_dtype)

        out = out.astype(orig_dtype)
        return out


jax_layer_norm2 = LayerNorm(96)


j_out2 = jax_layer_norm2(jnp.array(x))
print(j_out2.shape)

print(np.allclose(t_out, np.array(j_out2), atol=1e-5))
