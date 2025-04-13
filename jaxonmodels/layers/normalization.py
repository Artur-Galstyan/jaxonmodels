import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Hashable, Sequence
from equinox.nn import State
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions.utils import default_floating_dtype


class BatchNorm(eqx.nn.StatefulLayer):
    state_index: eqx.nn.StateIndex

    gamma: Float[Array, "size"] | None
    beta: Float[Array, "size"] | None

    inference: bool
    axis_name: Hashable | Sequence[Hashable]

    size: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    affine: bool = eqx.field(static=True)

    def __init__(
        self,
        size: int,
        axis_name: Hashable | Sequence[Hashable],
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        inference: bool = False,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.size = size
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.inference = inference
        self.axis_name = axis_name

        self.gamma = jnp.ones(self.size, dtype=dtype) if self.affine else None
        self.beta = jnp.zeros(self.size, dtype=dtype) if self.affine else None

        self.state_index = eqx.nn.StateIndex(
            (jnp.zeros(size, dtype=dtype), jnp.ones(size, dtype=dtype))
        )

    def __call__(
        self,
        x: Array,
        state: State,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, State]:
        running_mean, running_var = state.get(self.state_index)

        input_shape = x.shape
        ndim = len(input_shape)

        if ndim == 1:
            batch_mean = jax.lax.pmean(x, axis_name=self.axis_name)
            batch_size = jax.lax.psum(1, axis_name=self.axis_name)

            if self.inference:
                x_normalized = (x - running_mean) / jnp.sqrt(running_var + self.eps)
            else:
                xmu = x - batch_mean
                sq = xmu**2
                batch_var = jax.lax.pmean(sq, axis_name=self.axis_name)
                std = jnp.sqrt(batch_var + self.eps)
                x_normalized = xmu / std

                correction_factor = batch_size / jnp.maximum(batch_size - 1, 1)
                running_mean = (
                    1 - self.momentum
                ) * running_mean + self.momentum * batch_mean
                running_var = (1 - self.momentum) * running_var + self.momentum * (
                    batch_var * correction_factor
                )

                state = state.set(self.state_index, (running_mean, running_var))
        else:
            spatial_axes = tuple(range(1, ndim))  # All dims except channel dim (0)

            if self.inference:
                x_normalized = (
                    x - running_mean.reshape((-1,) + (1,) * (ndim - 1))
                ) / jnp.sqrt(running_var.reshape((-1,) + (1,) * (ndim - 1)) + self.eps)
            else:
                spatial_mean = jnp.mean(x, axis=spatial_axes)

                batch_mean = jax.lax.pmean(spatial_mean, axis_name=self.axis_name)
                batch_size = jax.lax.psum(1, axis_name=self.axis_name)

                broadcast_shape = (-1,) + (1,) * (ndim - 1)
                batch_mean_broadcasted = batch_mean.reshape(broadcast_shape)

                xmu = x - batch_mean_broadcasted
                sq = xmu**2

                spatial_var = jnp.mean(sq, axis=spatial_axes)
                batch_var = jax.lax.pmean(spatial_var, axis_name=self.axis_name)

                batch_var_broadcasted = batch_var.reshape(broadcast_shape)
                std = jnp.sqrt(batch_var_broadcasted + self.eps)

                x_normalized = xmu / std

                spatial_size = 1
                for dim in spatial_axes:
                    spatial_size *= x.shape[dim]
                total_size = batch_size * spatial_size

                correction_factor = total_size / jnp.maximum(total_size - 1, 1)
                running_mean = (
                    1 - self.momentum
                ) * running_mean + self.momentum * batch_mean
                running_var = (1 - self.momentum) * running_var + self.momentum * (
                    batch_var * correction_factor
                )

                state = state.set(self.state_index, (running_mean, running_var))

        out = x_normalized
        if self.affine and self.gamma is not None and self.beta is not None:
            if ndim > 1:
                broadcast_shape = (-1,) + (1,) * (ndim - 1)
                gamma_broadcasted = self.gamma.reshape(broadcast_shape)
                beta_broadcasted = self.beta.reshape(broadcast_shape)
                out = gamma_broadcasted * x_normalized + beta_broadcasted
            else:
                out = self.gamma * x_normalized + self.beta

        return out, state


class LocalResponseNormalization(eqx.Module):
    k: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __init__(self, k=2, n=5, alpha=1e-4, beta=0.75) -> None:
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        c, _, _ = x.shape
        p = jnp.pad(x, pad_width=[(self.n // 2, self.n // 2), (0, 0), (0, 0)])

        def _body(i):
            window = jax.lax.dynamic_slice_in_dim(p, i, self.n) ** 2
            d = (jnp.einsum("ijk->jk", window) * self.alpha + self.k) ** self.beta
            b = x[i] / d
            return b

        ys = eqx.filter_vmap(_body)(jnp.arange(c))
        return ys


class LayerNorm2d(eqx.Module):
    layer_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        shape: int | Sequence[int],
        eps: float = 0.00001,
        use_weight: bool = True,
        use_bias: bool = True,
        dtype: Any | None = None,
        *,
        elementwise_affine: bool | None = None,
    ):
        self.layer_norm = eqx.nn.LayerNorm(
            shape,
            eps,
            use_weight,
            use_bias,
            dtype=dtype,
            elementwise_affine=elementwise_affine,
        )

    def __call__(
        self, x: Float[Array, "h w c"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "h w c"]:
        x = eqx.filter_vmap(eqx.filter_vmap(self.layer_norm))(x)
        return x


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
