import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable, Sequence
from equinox.nn import LayerNorm
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions.utils import default_floating_dtype
from jaxonmodels.layers import ConvNormActivation, StochasticDepth
from jaxonmodels.layers.sequential import Sequential


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: int | None,
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class CNBlock(eqx.Module):
    layer_scale: Array
    dwconv: eqx.nn.Conv2d
    norm: eqx.Module
    pwconv1: eqx.nn.Linear
    pwconv2: eqx.nn.Linear
    stochastic_depth: StochasticDepth

    def __init__(
        self,
        dim: int,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer,
        *,
        key: PRNGKeyArray,
        dtype: Any,
    ) -> None:
        key, *keys = jax.random.split(key, 5)

        self.layer_scale = jnp.ones((dim, 1, 1), dtype=dtype) * layer_scale
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

        assert self.norm is not None

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
        residual = x
        x = self.dwconv(x)
        x = jnp.transpose(x, (1, 2, 0))
        x = eqx.filter_vmap(eqx.filter_vmap(self.norm))(x)  # pyright: ignore
        x = eqx.filter_vmap(eqx.filter_vmap(self.pwconv1))(x)
        x = jax.nn.gelu(x)
        x = eqx.filter_vmap(eqx.filter_vmap(self.pwconv2))(x)
        x = jnp.transpose(x, (2, 0, 1))
        x = self.layer_scale * x
        x = self.stochastic_depth(x, key=key, inference=inference)
        x = x + residual
        return x


class ConvNeXt(eqx.Module):
    features: Sequential
    avgpool: eqx.nn.AdaptiveAvgPool2d

    def __init__(
        self,
        block_setting: list[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Callable[..., eqx.Module] | None = None,
        norm_layer: Callable[..., eqx.Module] | None = None,
        *,
        key: PRNGKeyArray,
        dtype: Any,
    ) -> None:
        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (
            isinstance(block_setting, Sequence)
            and all([isinstance(s, CNBlockConfig) for s in block_setting])
        ):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = LayerNorm
        assert norm_layer is not None
        layers: list[eqx.Module] = []

        key, subkey = jax.random.split(key)
        firstconv_output_channels = block_setting[0].input_channels

        layers.append(
            ConvNormActivation(
                2,
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                use_bias=True,
                key=subkey,
                dtype=dtype,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: list[eqx.Module] = []
            key, *subkeys = jax.random.split(key, cnf.num_layers)
            for i in range(cnf.num_layers):
                sd_prob = (
                    stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                )
                stage.append(
                    block(
                        cnf.input_channels,
                        layer_scale,
                        sd_prob,
                        norm_layer,
                        key=subkeys[i],
                        dtype=dtype,
                    )
                )
                stage_block_id += 1
            layers.append(Sequential(stage))
            if cnf.out_channels is not None:
                # Downsampling
                key, subkey = jax.random.split(key)
                layers.append(
                    Sequential(
                        [
                            norm_layer(cnf.input_channels),
                            eqx.nn.Conv2d(
                                cnf.input_channels,
                                cnf.out_channels,
                                kernel_size=2,
                                stride=2,
                                dtype=dtype,
                                key=subkey,
                            ),
                        ]
                    )
                )

        self.features = Sequential(layers)
        self.avgpool = eqx.nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels
            if lastblock.out_channels is not None
            else lastblock.input_channels
        )
        key, subkey = jax.random.split(key)
        self.classifier = Sequential(
            [
                norm_layer(lastconv_output_channels, dtype=dtype),
                eqx.nn.Lambda(lambda x: jnp.ravel(x)),
                eqx.nn.Linear(
                    lastconv_output_channels, num_classes, key=subkey, dtype=dtype
                ),
            ]
        )

        # todo: init properly
