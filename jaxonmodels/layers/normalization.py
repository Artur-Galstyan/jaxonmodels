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
        inference: bool | None = None,
    ) -> tuple[Array, State]:
        if inference is None:
            inference = self.inference

        running_mean, running_var = state.get(self.state_index)

        input_shape = x.shape
        ndim = len(input_shape)

        if ndim == 1:
            batch_mean = jax.lax.pmean(x, axis_name=self.axis_name)
            batch_size = jax.lax.psum(1, axis_name=self.axis_name)

            if inference:
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

            if inference:
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
