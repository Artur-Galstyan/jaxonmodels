import functools
import os
from urllib.request import urlretrieve

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable, Literal, Sequence
from equinox.nn import StatefulLayer
from jaxonlayers.layers import (
    BatchedLinear,
    ConvNormActivation,
    LayerNorm,
    StochasticDepth,
)
from jaxonlayers.layers.abstract import AbstractNorm
from jaxtyping import Array, Float, PRNGKeyArray
from statedict2pytree.converter import autoconvert

from jaxonmodels.functions import default_floating_dtype, dtype_to_str
from jaxonmodels.functions.utils import get_cache_path

_MODELS = {
    "convnext_tiny_IMAGENET1K_V1": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
    "convnext_small_IMAGENET1K_V1": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
    "convnext_base_IMAGENET1K_V1": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
    "convnext_large_IMAGENET1K_V1": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
}


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
    block: eqx.nn.Sequential
    layer_scale: Array
    stochastic_depth: StochasticDepth

    def __init__(
        self,
        dim: int,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., AbstractNorm] | None,
        *,
        key: PRNGKeyArray,
        dtype: Any,
    ) -> None:
        key, *keys = jax.random.split(key, 5)

        self.block = eqx.nn.Sequential(
            layers=[
                eqx.nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=7,
                    padding=3,
                    groups=dim,
                    use_bias=True,
                    key=keys[0],
                    dtype=dtype,
                ),
                eqx.nn.Lambda(fn=lambda x: jnp.transpose(x, (1, 2, 0))),
                LayerNorm(shape=dim, eps=1e-6)
                if norm_layer is None
                else norm_layer(shape=dim, eps=1e-6),
                BatchedLinear(
                    in_features=dim,
                    out_features=4 * dim,
                    use_bias=True,
                    key=keys[2],
                    dtype=dtype,
                ),
                eqx.nn.Lambda(fn=lambda x: jax.nn.gelu(x)),
                BatchedLinear(
                    in_features=4 * dim,
                    out_features=dim,
                    use_bias=True,
                    key=keys[3],
                    dtype=dtype,
                ),
                eqx.nn.Lambda(fn=lambda x: jnp.transpose(x, (2, 0, 1))),
            ]
        )

        self.layer_scale = jnp.ones((dim, 1, 1), dtype=dtype) * layer_scale

        self.stochastic_depth = StochasticDepth(p=stochastic_depth_prob, mode="row")

    def __call__(
        self,
        x: Float[Array, "c h w"],
        key: PRNGKeyArray,
    ) -> Float[Array, "c h w"]:
        residual = x
        x = self.block(x)
        x = self.layer_scale * x
        x = self.stochastic_depth(x, key=key)
        x = x + residual
        return x


class ConvNeXt(StatefulLayer):
    features: eqx.nn.Sequential
    avgpool: eqx.nn.AdaptiveAvgPool2d
    classifier: eqx.nn.Sequential

    inference: bool

    def __init__(
        self,
        block_setting: list[CNBlockConfig],
        stochastic_depth_prob: float,
        layer_scale: float,
        num_classes: int,
        block: Callable[..., eqx.Module] | None,
        norm_layer: Callable[..., AbstractNorm] | None,
        inference: bool,
        key: PRNGKeyArray,
        dtype: Any,
    ) -> None:
        self.inference = inference
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

        key, subkey = jax.random.split(key)
        firstconv_output_channels = block_setting[0].input_channels

        if norm_layer is None:
            norm_layer = functools.partial(LayerNorm, dtype=dtype)
        assert norm_layer is not None
        layers: list[eqx.Module] = []

        # Stem: patchify conv followed by LayerNorm in channels-last format
        layers.append(
            eqx.nn.Sequential(
                [
                    ConvNormActivation(
                        2,
                        3,
                        firstconv_output_channels,
                        kernel_size=4,
                        stride=4,
                        padding=0,
                        norm_layer=None,
                        activation_layer=None,
                        use_bias=True,
                        key=subkey,
                        dtype=dtype,
                    ),
                    eqx.nn.Lambda(fn=lambda x: jnp.transpose(x, (1, 2, 0))),
                    norm_layer(shape=firstconv_output_channels),  # pyright: ignore
                    eqx.nn.Lambda(fn=lambda x: jnp.transpose(x, (2, 0, 1))),
                ]
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: list[eqx.Module] = []
            key, *subkeys = jax.random.split(key, cnf.num_layers + 5)
            for i in range(cnf.num_layers):
                sd_prob = (
                    stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                )
                stage.append(
                    block(
                        cnf.input_channels,
                        layer_scale,
                        sd_prob,
                        norm_layer=None,
                        key=subkeys[i],
                        dtype=dtype,
                    )
                )
                stage_block_id += 1
            layers.append(eqx.nn.Sequential(stage))  # ty:ignore[invalid-argument-type]
            if cnf.out_channels is not None:
                # Downsampling: LayerNorm in channels-last then strided conv
                key, subkey = jax.random.split(key)
                layers.append(
                    eqx.nn.Sequential(
                        [
                            eqx.nn.Lambda(fn=lambda x: jnp.transpose(x, (1, 2, 0))),
                            norm_layer(cnf.input_channels),  # pyright: ignore
                            eqx.nn.Lambda(fn=lambda x: jnp.transpose(x, (2, 0, 1))),
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

        self.features = eqx.nn.Sequential(layers)  # ty:ignore[invalid-argument-type]
        self.avgpool = eqx.nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels
            if lastblock.out_channels is not None
            else lastblock.input_channels
        )
        key, subkey = jax.random.split(key)
        self.classifier = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(fn=lambda x: jnp.transpose(x, (1, 2, 0))),
                norm_layer(lastconv_output_channels),  # pyright: ignore
                eqx.nn.Lambda(fn=lambda x: jnp.transpose(x, (2, 0, 1))),
                eqx.nn.Lambda(lambda x: jnp.ravel(x)),
                eqx.nn.Linear(
                    lastconv_output_channels, num_classes, key=subkey, dtype=dtype
                ),
            ]
        )

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ):
        key, *subkeys = jax.random.split(key, 4)
        x, state = self.features(x, key=subkeys[0], state=state)
        x = self.avgpool(x, key=subkeys[1])
        x, state = self.classifier(x, key=subkeys[2], state=state)
        return x, state

    @staticmethod
    def with_weights(
        model: Literal[
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
        ],
        key: PRNGKeyArray | None = None,
        dtype: Any | None = None,
        **model_kwargs,
    ) -> tuple["ConvNeXt", eqx.nn.State]:
        """
        Load a ConvNeXt variant with ImageNet-1K pretrained weights.

        Downloads PyTorch weights, converts them to JAX format via autoconvert,
        and caches the result as an .eqx file so subsequent calls are instant.

        Args:
            model: Which variant to load — "convnext_tiny", "convnext_small",
                "convnext_base", or "convnext_large".
            key: PRNG key for initialising the model skeleton. Defaults to
                jax.random.key(42).
            dtype: Parameter dtype. Defaults to
                jaxonmodels.functions.default_floating_dtype().
            **model_kwargs: Forwarded to the variant constructor. Supported:
                inference (bool, default False), num_classes (int, default 1000).

        Returns:
            A (model, state) tuple.
        """
        if "inference" not in model_kwargs:
            model_kwargs["inference"] = False
        if key is None:
            key = jax.random.key(42)

        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None

        weights_key: str
        match model:
            case "convnext_tiny":
                convnext_model, state = _convnext_tiny(key, dtype=dtype, **model_kwargs)
                weights_key = "convnext_tiny_IMAGENET1K_V1"
            case "convnext_small":
                convnext_model, state = _convnext_small(
                    key, dtype=dtype, **model_kwargs
                )
                weights_key = "convnext_small_IMAGENET1K_V1"
            case "convnext_base":
                convnext_model, state = _convnext_base(key, dtype=dtype, **model_kwargs)
                weights_key = "convnext_base_IMAGENET1K_V1"
            case "convnext_large":
                convnext_model, state = _convnext_large(
                    key, dtype=dtype, **model_kwargs
                )
                weights_key = "convnext_large_IMAGENET1K_V1"
            case _:
                raise ValueError(f"Unknown model: {model!r}")

        dtype_str = dtype_to_str(dtype)
        jaxonmodels_path = get_cache_path("convnext")
        cache_file = str(jaxonmodels_path / f"convnext-{model}-{dtype_str}.eqx")

        if os.path.exists(cache_file):
            return eqx.tree_deserialise_leaves(cache_file, (convnext_model, state))

        weights_url = _MODELS[weights_key]
        weights_file = os.path.join(jaxonmodels_path, f"{weights_key}.pth")
        if not os.path.exists(weights_file):
            urlretrieve(weights_url, weights_file)

        import torch

        weights_dict = torch.load(
            weights_file, map_location=torch.device("cpu"), weights_only=True
        )

        convnext_model, state = autoconvert(
            (convnext_model, state), weights_dict, dtype=dtype
        )

        eqx.tree_serialise_leaves(cache_file, (convnext_model, state))

        return convnext_model, state


def _convnext_tiny(key, inference: bool, num_classes=1000, dtype: Any = None):
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]

    return eqx.nn.make_with_state(ConvNeXt)(
        block_setting=block_setting,
        stochastic_depth_prob=0.1,
        num_classes=num_classes,
        layer_scale=1e-6,
        block=None,
        norm_layer=None,
        inference=inference,
        key=key,
        dtype=dtype,
    )


def _convnext_small(key, inference: bool, num_classes=1000, dtype: Any = None):
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]

    return eqx.nn.make_with_state(ConvNeXt)(
        block_setting=block_setting,
        stochastic_depth_prob=0.4,
        num_classes=num_classes,
        layer_scale=1e-6,
        block=None,
        norm_layer=None,
        inference=inference,
        key=key,
        dtype=dtype,
    )


def _convnext_base(key, inference: bool, num_classes=1000, dtype: Any = None):
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None

    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]

    return eqx.nn.make_with_state(ConvNeXt)(
        block_setting=block_setting,
        stochastic_depth_prob=0.5,
        num_classes=num_classes,
        layer_scale=1e-6,
        block=None,
        norm_layer=None,
        inference=inference,
        key=key,
        dtype=dtype,
    )


def _convnext_large(key, inference: bool, num_classes=1000, dtype: Any = None):
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None

    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]

    return eqx.nn.make_with_state(ConvNeXt)(
        block_setting=block_setting,
        stochastic_depth_prob=0.5,
        num_classes=num_classes,
        layer_scale=1e-6,
        block=None,
        norm_layer=None,
        inference=inference,
        key=key,
        dtype=dtype,
    )
