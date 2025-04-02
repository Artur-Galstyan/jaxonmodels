import os
from pathlib import Path
from urllib.request import urlretrieve

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable, Literal, Sequence
from equinox.nn import LayerNorm
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions import default_floating_dtype, dtype_to_str
from jaxonmodels.layers import ConvNormActivation, StochasticDepth
from jaxonmodels.statedict2pytree.s2p import (
    convert,
    move_running_fields_to_the_end,
    pytree_to_fields,
    serialize_pytree,
    state_dict_to_fields,
)

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
    features: eqx.nn.Sequential
    avgpool: eqx.nn.AdaptiveAvgPool2d
    classifier: eqx.nn.Sequential

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
                        norm_layer,
                        key=subkeys[i],
                        dtype=dtype,
                    )
                )
                stage_block_id += 1
            layers.append(eqx.nn.Sequential(stage))  # pyright: ignore
            if cnf.out_channels is not None:
                # Downsampling
                key, subkey = jax.random.split(key)
                layers.append(
                    eqx.nn.Sequential(
                        [
                            norm_layer(cnf.input_channels),  # pyright: ignore
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

        self.features = eqx.nn.Sequential(layers)  # pyright: ignore
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
                norm_layer(lastconv_output_channels, dtype=dtype),  # pyright: ignore
                eqx.nn.Lambda(lambda x: jnp.ravel(x)),
                eqx.nn.Linear(
                    lastconv_output_channels, num_classes, key=subkey, dtype=dtype
                ),
            ]
        )

        # todo: init properly

    def __call__(
        self,
        x: Array,
        key: PRNGKeyArray,
        *,
        state: eqx.nn.State | None = None,
    ):
        # todo: Handle state if norm_layer is stateful
        key, *subkeys = jax.random.split(key, 3)
        x = self.features(x, key=subkeys[0])
        x = self.avgpool(x, key=subkeys[1])
        x = self.classifier(x, key=subkeys[2])
        return x


def _convnext_tiny(key, num_classes=1000, dtype: Any = None):
    """Creates a ConvNeXt Tiny model."""
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]

    model = ConvNeXt(
        block_setting=block_setting,
        stochastic_depth_prob=0.1,
        num_classes=num_classes,
        key=key,
        dtype=dtype,
    )

    return model


def _convnext_small(key, num_classes=1000, dtype: Any = None):
    """Creates a ConvNeXt Small model."""
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]

    model = ConvNeXt(
        block_setting=block_setting,
        stochastic_depth_prob=0.4,
        num_classes=num_classes,
        key=key,
        dtype=dtype,
    )

    return model


def _convnext_base(key, num_classes=1000, dtype: Any = None):
    """Creates a ConvNeXt Base model."""
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None

    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]

    model = ConvNeXt(
        block_setting=block_setting,
        stochastic_depth_prob=0.5,
        num_classes=num_classes,
        key=key,
        dtype=dtype,
    )

    return model


def _convnext_large(key, num_classes=1000, dtype: Any = None):
    """Creates a ConvNeXt Large model."""
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None

    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]

    model = ConvNeXt(
        block_setting=block_setting,
        stochastic_depth_prob=0.5,
        num_classes=num_classes,
        key=key,
        dtype=dtype,
    )

    return model


def _with_weights(
    model,
    weights: str,
    cache: bool,
    dtype: Any,
):
    """Load pre-trained weights for the model."""
    dtype_str = dtype_to_str(dtype)

    weights_url = _MODELS.get(weights)
    if weights_url is None:
        raise ValueError(f"No weights found for {weights}")

    # Create directories for caching
    jaxonmodels_dir = os.path.expanduser("~/.jaxonmodels/models")
    os.makedirs(jaxonmodels_dir, exist_ok=True)

    # Define cache file path including dtype
    cache_filename = f"{weights}-{dtype_str}.eqx"
    cache_filepath = str(Path(jaxonmodels_dir) / cache_filename)

    # Check if cached model exists
    if cache:
        if os.path.exists(cache_filepath):
            return eqx.tree_deserialise_leaves(
                cache_filepath,
                model,
            )

    # Download weights if not already downloaded
    weights_dir = os.path.expanduser("~/.jaxonmodels/pytorch_weights")
    os.makedirs(weights_dir, exist_ok=True)
    filename = weights_url.split("/")[-1]
    weights_file = os.path.join(weights_dir, filename)
    if not os.path.exists(weights_file):
        print(f"Downloading weights from {weights_url} to {weights_file}")
        urlretrieve(weights_url, weights_file)
    else:
        print(f"Using existing weights file: {weights_file}")

    # Load PyTorch weights
    import torch

    print("Loading PyTorch state dict...")
    weights_dict = torch.load(
        weights_file,
        map_location=torch.device("cpu"),  # weights_only=True removed for compatibility
    )
    # If it's a checkpoint dict, extract the state_dict
    if not isinstance(weights_dict, dict):
        # Try loading as a JIT model like in CLIP if simple load fails
        try:
            torch_model = torch.jit.load(weights_file, map_location=torch.device("cpu"))
            temp_dict = {}
            for name, param in torch_model.named_parameters():
                temp_dict[name] = param.clone().detach()
            for name, buffer in torch_model.named_buffers():
                temp_dict[name] = buffer.clone().detach()
            weights_dict = temp_dict
            print("Loaded weights from TorchScript model.")
        except Exception as e:
            raise TypeError(
                f"Could not load weights. "
                f"Expected state_dict but got {type(weights_dict)}. "
                f"JIT load failed with: {e}"
            )
    elif "state_dict" in weights_dict:
        weights_dict = weights_dict["state_dict"]

    print("Converting PyTorch weights to JAX format...")
    # Convert PyTorch weights to JAX format
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    jaxfields, state_indices = pytree_to_fields(model)

    model = convert(
        weights_dict,
        model,
        jaxfields,
        state_indices,
        torchfields,
        dtype=dtype,
    )

    if cache:
        serialize_pytree(model, cache_filepath)

    return model


def _assert_model_and_weights_fit(
    model: Literal[
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
    ],
    weights: Literal[
        "convnext_tiny_IMAGENET1K_V1",
        "convnext_small_IMAGENET1K_V1",
        "convnext_base_IMAGENET1K_V1",
        "convnext_large_IMAGENET1K_V1",
    ],
):
    """Ensures the model and weights are compatible."""
    if "_" in weights:
        weights_model = weights.split("_IMAGENET")[0].lower()

        # Check if the specified weights are compatible with the model
        if weights_model != model:
            raise ValueError(
                f"Model {model} is incompatible with weights {weights}. "
                f"Expected weights starting with '{model}'."
            )


def load_convnext(
    model: Literal[
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
    ],
    weights: Literal[
        "convnext_tiny_IMAGENET1K_V1",
        "convnext_small_IMAGENET1K_V1",
        "convnext_base_IMAGENET1K_V1",
        "convnext_large_IMAGENET1K_V1",
    ]
    | None = None,
    num_classes: int = 1000,
    cache: bool = True,
    *,
    key: PRNGKeyArray | None = None,
    dtype: Any = None,
) -> ConvNeXt:
    """
    Load a ConvNeXt model with optional pre-trained weights.

    Args:
        model: The name of the ConvNeXt model to load
        weights: The name of the weights to load (from _MODELS), or None for random init
        num_classes: Number of output classes
        cache: Whether to cache the model and weights.
        key: Random key for initialization. Defaults to jax.random.key(42).
        dtype: The data type for the model's parameters (e.g., jnp.float32).
               Defaults to jaxonmodels.functions.default_floating_dtype().

    Returns:
        A ConvNeXt model
    """
    if key is None:
        key = jax.random.key(42)

    # Determine default dtype if not provided before passing to constructors
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None  # Ensure dtype is set

    convnext = None

    match model:
        case "convnext_tiny":
            convnext = _convnext_tiny(key, num_classes, dtype=dtype)
        case "convnext_small":
            convnext = _convnext_small(key, num_classes, dtype=dtype)
        case "convnext_base":
            convnext = _convnext_base(key, num_classes, dtype=dtype)
        case "convnext_large":
            convnext = _convnext_large(key, num_classes, dtype=dtype)
        case _:
            raise ValueError(f"Unknown model name: {model}")

    if weights:
        _assert_model_and_weights_fit(model, weights)
        convnext = _with_weights(convnext, weights, cache, dtype=dtype)

    assert convnext is not None

    return convnext
