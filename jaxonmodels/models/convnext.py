import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from equinox.nn import LayerNorm
from jaxtyping import Array, Float, PRNGKeyArray, Shaped

from jaxonmodels.layers import StochasticDepth


class CNBlock(eqx.Module):
    layer_scale: Shaped[Array, "dim 1 1"]
    dwconv: eqx.nn.Conv2d
    norm: eqx.Module
    pwconv1: eqx.nn.Linear
    pwconv2: eqx.nn.Linear
    stochastic_depth: StochasticDepth

    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float,
        stochastic_depth_prob: float,
        norm_layer,
        *,
        key: PRNGKeyArray,
        dtype: Any,
    ) -> None:
        key, *keys = jax.random.split(key, 5)

        self.layer_scale = jnp.ones((dim, 1, 1), dtype=dtype) * layer_scale_init_value
        self.dwconv = eqx.nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim,
            use_bias=True,
            key=keys[0],
            dtype=dtype,
        )
        if norm_layer is None:
            self.norm = LayerNorm(shape=dim, eps=1e-6)
        else:
            self.norm = norm_layer(shape=dim, eps=1e-6)

        self.pwconv1 = eqx.nn.Linear(
            in_features=dim,
            out_features=4 * dim,
            use_bias=True,
            key=keys[2],
            dtype=dtype,
        )

        self.pwconv2 = eqx.nn.Linear(
            in_features=4 * dim,
            out_features=dim,
            use_bias=True,
            key=keys[3],
            dtype=dtype,
        )
        self.stochastic_depth = StochasticDepth(p=stochastic_depth_prob, mode="row")

    def __call__(
        self,
        x: Float[Array, "c h w"],
        inference: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "c h w"]:
        print("JAX START")
        print(x.shape)
        residual = x
        x = self.dwconv(x)
        x = jnp.transpose(x, (1, 2, 0))
        print(x.shape)
        x = eqx.filter_vmap(eqx.filter_vmap(self.norm))(x)  # pyright: ignore
        print(x.shape)
        x = self.dwconv(x)
        print(x.shape)
        x = eqx.filter_vmap(self.norm)(x)  # pyright: ignore
        print(x.shape)
        x = self.pwconv1(x)
        print(x.shape)
        x = jax.nn.gelu(x)
        print(x.shape)
        x = self.pwconv2(x)
        print(x.shape)

        x = self.layer_scale * x
        print(x.shape)

        x = self.stochastic_depth(x, key=key, inference=inference)
        print(x.shape)

        x = x + residual
        print(x.shape)
        print("JAX END")
        return x
