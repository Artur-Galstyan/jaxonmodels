import copy
import functools
import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable, Literal, Sequence
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions import (
    make_divisible,
    make_ntuple,
)
from jaxonmodels.layers import BatchNorm, SqueezeExcitation, StochasticDepth
from jaxonmodels.statedict2pytree.s2p import (
    convert,
    move_running_fields_to_the_end,
    pytree_to_fields,
    serialize_pytree,
    state_dict_to_fields,
)

_MODELS = {
    "efficientnet_b0_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
    "efficientnet_b1_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287d4.pth",
    "efficientnet_b1_IMAGENET1K_V2": "https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth",
    "efficientnet_b2_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth",
    "efficientnet_b3_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth",
    "efficientnet_b4_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth",
    "efficientnet_b5_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth",
    "efficientnet_b6_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-24a108a5.pth",
    "efficientnet_b7_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth",
    "efficientnet_v2_s_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
    "efficientnet_v2_m_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
    "efficientnet_v2_l_IMAGENET1K_V1": "https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth",
}


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., "MBConv | FusedMBConv"]

    @staticmethod
    def adjust_channels(
        channels: int, width_mult: float, min_value: int | None = None
    ) -> int:
        return make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper &
    # Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            MBConv,
        )

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            FusedMBConv,
        )


class Conv2dNormActivation(eqx.Module):
    conv2d: eqx.nn.Conv2d
    norm: eqx.Module | None
    activation: eqx.nn.Lambda | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] | None = None,
        groups: int = 1,
        norm_layer: Callable[..., eqx.Module] | None = BatchNorm,
        activation_layer: Callable[..., Array] | None = jax.nn.relu,
        dilation: int | Sequence[int] = 1,
        use_bias: bool | None = None,
        dtype=None,
        axis_name: str = "batch",
        *,
        key: PRNGKeyArray,
    ):
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = (
                    len(kernel_size)
                    if isinstance(kernel_size, Sequence)
                    else len(dilation)  # pyright: ignore
                )
                kernel_size = make_ntuple(kernel_size, _conv_dim)
                dilation = make_ntuple(dilation, _conv_dim)
                padding = tuple(
                    (kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim)
                )
        if use_bias is None:
            use_bias = norm_layer is None

        key, subkey = jax.random.split(key)

        self.conv2d = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            dtype=dtype,
            key=subkey,
        )

        if norm_layer is not None:
            self.norm = norm_layer(out_channels, axis_name=axis_name)

        if activation_layer is not None:
            self.activation = eqx.nn.Lambda(activation_layer)
        else:
            self.activation = None

    def __call__(
        self,
        x: Float[Array, "c h w"],
        state: eqx.nn.State,
        inference: bool,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "c_out h_out w_out"], eqx.nn.State]:
        x = self.conv2d(x)

        if self.norm:
            x, state = self.norm(x, state, inference=inference)  # pyright: ignore

        if self.activation:
            x = self.activation(x)

        return x, state


class MBConv(eqx.Module):
    use_res_connect: bool = eqx.field(static=True)

    expand_conv2d: Conv2dNormActivation | None
    depthwise_conv2d: Conv2dNormActivation
    se_layer: SqueezeExcitation

    project_conv2d: Conv2dNormActivation
    stochastic_depth: StochasticDepth

    def __init__(
        self,
        cnf,
        stochastic_depth_prob: float,
        key: PRNGKeyArray,
    ):
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        activation_layer = jax.nn.silu

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            key, subkey = jax.random.split(key)
            self.expand_conv2d = Conv2dNormActivation(
                cnf.input_channels,
                expanded_channels,
                kernel_size=1,
                norm_layer=BatchNorm,
                activation_layer=activation_layer,
                key=subkey,
            )
        else:
            self.expand_conv2d = None

        # depthwise
        key, subkey = jax.random.split(key)
        self.depthwise_conv2d = Conv2dNormActivation(
            expanded_channels,
            expanded_channels,
            kernel_size=cnf.kernel,
            stride=cnf.stride,
            groups=expanded_channels,
            norm_layer=BatchNorm,
            activation_layer=activation_layer,
            key=subkey,
        )
        squeeze_channels = max(1, cnf.input_channels // 4)

        key, subkey = jax.random.split(key)
        self.se_layer = SqueezeExcitation(
            expanded_channels, squeeze_channels, key=subkey
        )

        # project
        key, subkey = jax.random.split(key)
        self.project_conv2d = Conv2dNormActivation(
            expanded_channels,
            cnf.out_channels,
            kernel_size=1,
            norm_layer=BatchNorm,
            activation_layer=None,
            key=subkey,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def __call__(
        self, x: Array, state: eqx.nn.State, inference: bool, key: PRNGKeyArray
    ):
        if self.expand_conv2d:
            result, state = self.expand_conv2d(x, state, inference)
        else:
            result = x
        result, state = self.depthwise_conv2d(result, state, inference)
        result = self.se_layer(result, activation=jax.nn.silu)

        result, state = self.project_conv2d(result, state, inference)

        if self.use_res_connect:
            result = self.stochastic_depth(result, inference=inference, key=key)
            result += x
        return result, state


class FusedMBConv(eqx.Module):
    fused_expand: Conv2dNormActivation
    project: Conv2dNormActivation | None
    stochastic_depth: StochasticDepth

    use_res_connect: bool = eqx.field(static=True)

    def __init__(
        self,
        cnf,
        stochastic_depth_prob: float,
        key: PRNGKeyArray,
    ):
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            key, subkey, subkey2 = jax.random.split(key, 3)
            self.fused_expand = Conv2dNormActivation(
                cnf.input_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                norm_layer=BatchNorm,
                activation_layer=jax.nn.silu,
                key=subkey,
            )

            # project
            self.project = Conv2dNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=BatchNorm,
                activation_layer=None,
                key=subkey2,
            )
        else:
            key, subkey = jax.random.split(key)
            self.fused_expand = Conv2dNormActivation(
                cnf.input_channels,
                cnf.out_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                norm_layer=BatchNorm,
                activation_layer=jax.nn.silu,
                key=subkey,
            )
            self.project = None

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def __call__(
        self, x: Array, state: eqx.nn.State, inference: bool, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        result, state = self.fused_expand(x, state, inference)

        if self.project:
            result, state = self.project(result, state, inference)

        if self.use_res_connect:
            result = self.stochastic_depth(result, inference, key)
            result += x
        return result, state


class InvertedResidualBlock(eqx.Module):
    layers: list[MBConv | FusedMBConv]

    def __init__(self, layers: list[MBConv | FusedMBConv]):
        self.layers = layers

    def __call__(
        self, x: Array, state: eqx.nn.State, inference: bool, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        for stage in self.layers:
            key, subkey = jax.random.split(key)
            x, state = stage(x, state, inference, subkey)

        return x, state


class Classifier(eqx.Module):
    dropout: eqx.nn.Dropout
    linear: eqx.nn.Linear

    def __init__(
        self, p: float, in_features: int, out_features: int, key: PRNGKeyArray
    ):
        self.dropout = eqx.nn.Dropout(p=p)
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)

    def __call__(self, x: Array, inference: bool, key: PRNGKeyArray) -> Array:
        x = self.dropout(x, inference=inference, key=key)
        x = self.linear(x)

        return x


class EfficientNet(eqx.Module):
    features: list[Conv2dNormActivation | MBConv | FusedMBConv | InvertedResidualBlock]
    avgpool: eqx.nn.AdaptiveAvgPool2d
    classifier: Classifier

    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        n_classes: int = 1000,
        last_channel: int | None = None,
        *,
        key: PRNGKeyArray,
    ):
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
            )

        key, *subkeys = jax.random.split(key, 10)

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.features = []
        self.features.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=BatchNorm,
                activation_layer=jax.nn.silu,
                key=subkeys[0],
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: list[MBConv | FusedMBConv] = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)

                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )
                key, subkey = jax.random.split(key)
                stage.append(block_cnf.block(block_cnf, sd_prob, subkey))
                stage_block_id += 1

            self.features.append(InvertedResidualBlock(stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        key, subkey = jax.random.split(key)
        self.features.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=BatchNorm,
                activation_layer=jax.nn.silu,
                key=subkey,
            )
        )

        self.avgpool = eqx.nn.AdaptiveAvgPool2d(1)
        key, subkey = jax.random.split(key)
        self.classifier = Classifier(
            dropout, lastconv_output_channels, n_classes, subkey
        )
        # todo: kaiming init

    def __call__(
        self, x: Array, state: eqx.nn.State, inference: bool, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        for feature in self.features:
            key, subkey = jax.random.split(key)
            x, state = feature(x, state, inference, key)

        x = self.avgpool(x)
        x = jnp.ravel(x)

        key, subkey = jax.random.split(key)
        x = self.classifier(x, inference, key)

        return x, state


def _efficientnet(
    inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig],
    dropout: float,
    last_channel: int | None,
    n_classes: int,
    key: PRNGKeyArray,
) -> tuple[EfficientNet, eqx.nn.State]:
    model, state = eqx.nn.make_with_state(EfficientNet)(
        inverted_residual_setting,
        dropout,
        last_channel=last_channel,
        n_classes=n_classes,
        key=key,
    )

    return model, state


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> tuple[Sequence[MBConvConfig | FusedMBConvConfig], int | None]:
    inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig]
    if arch.startswith("efficientnet_b"):
        bneck_conf = functools.partial(
            MBConvConfig,
            width_mult=kwargs.pop("width_mult"),
            depth_mult=kwargs.pop("depth_mult"),
        )
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def _with_weights(
    pytree,
    weights: str,
    cache: bool,
):
    """Load pre-trained weights for the model"""
    efficientnet, state = pytree

    weights_url = _MODELS.get(weights)
    if weights_url is None:
        raise ValueError(f"No weights found for {weights}")

    # Create directories for caching
    jaxonmodels_dir = os.path.expanduser("~/.jaxonmodels/models")
    os.makedirs(jaxonmodels_dir, exist_ok=True)

    # Check if cached model exists
    if cache:
        if os.path.exists(str(Path(jaxonmodels_dir) / f"{weights}.eqx")):
            return eqx.tree_deserialise_leaves(
                str(Path(jaxonmodels_dir) / f"{weights}.eqx"),
                (efficientnet, state),
            )

    # Download weights if not already downloaded
    weights_dir = os.path.expanduser("~/.jaxonmodels/pytorch_weights")
    os.makedirs(weights_dir, exist_ok=True)
    filename = weights_url.split("/")[-1]
    weights_file = os.path.join(weights_dir, filename)
    if not os.path.exists(weights_file):
        urlretrieve(weights_url, weights_file)

    # Load PyTorch weights
    import torch

    weights_dict = torch.load(
        weights_file, map_location=torch.device("cpu"), weights_only=True
    )

    # Convert PyTorch weights to JAX format
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    jaxfields, state_indices = pytree_to_fields((efficientnet, state))

    efficientnet, state = convert(
        weights_dict, (efficientnet, state), jaxfields, state_indices, torchfields
    )

    # Cache the model if requested
    if cache:
        serialize_pytree(
            (efficientnet, state), str(Path(jaxonmodels_dir) / f"{weights}.eqx")
        )

    return efficientnet, state


def _efficientnet_b0(key, n_classes=1000):
    """Create EfficientNet B0 model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b0", width_mult=1.0, depth_mult=1.0
    )
    return _efficientnet(
        inverted_residual_setting, 0.2, last_channel, n_classes=n_classes, key=key
    )


def _efficientnet_b1(key, n_classes=1000):
    """Create EfficientNet B1 model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b1", width_mult=1.0, depth_mult=1.1
    )
    return _efficientnet(
        inverted_residual_setting, 0.2, last_channel, n_classes=n_classes, key=key
    )


def _efficientnet_b2(key, n_classes=1000):
    """Create EfficientNet B2 model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b2", width_mult=1.1, depth_mult=1.2
    )
    return _efficientnet(
        inverted_residual_setting, 0.3, last_channel, n_classes=n_classes, key=key
    )


def _efficientnet_b3(key, n_classes=1000):
    """Create EfficientNet B3 model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b3", width_mult=1.2, depth_mult=1.4
    )
    return _efficientnet(
        inverted_residual_setting, 0.3, last_channel, n_classes=n_classes, key=key
    )


def _efficientnet_b4(key, n_classes=1000):
    """Create EfficientNet B4 model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b4", width_mult=1.4, depth_mult=1.8
    )
    return _efficientnet(
        inverted_residual_setting, 0.4, last_channel, n_classes=n_classes, key=key
    )


def _efficientnet_b5(key, n_classes=1000):
    """Create EfficientNet B5 model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b5", width_mult=1.6, depth_mult=2.2
    )
    return _efficientnet(
        inverted_residual_setting, 0.4, last_channel, n_classes=n_classes, key=key
    )


def _efficientnet_b6(key, n_classes=1000):
    """Create EfficientNet B6 model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b6", width_mult=1.8, depth_mult=2.6
    )
    return _efficientnet(
        inverted_residual_setting, 0.5, last_channel, n_classes=n_classes, key=key
    )


def _efficientnet_b7(key, n_classes=1000):
    """Create EfficientNet B7 model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b7", width_mult=2.0, depth_mult=3.1
    )
    return _efficientnet(
        inverted_residual_setting, 0.5, last_channel, n_classes=n_classes, key=key
    )


def _efficientnet_v2_s(key, n_classes=1000):
    """Create EfficientNet V2 Small model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting, 0.2, last_channel, n_classes=n_classes, key=key
    )


def _efficientnet_v2_m(key, n_classes=1000):
    """Create EfficientNet V2 Medium model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return _efficientnet(
        inverted_residual_setting, 0.3, last_channel, n_classes=n_classes, key=key
    )


def _efficientnet_v2_l(key, n_classes=1000):
    """Create EfficientNet V2 Large model with random initialization"""
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return _efficientnet(
        inverted_residual_setting, 0.4, last_channel, n_classes=n_classes, key=key
    )


def _assert_model_and_weights_fit(
    model: Literal[
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
        "efficientnet_v2_s",
        "efficientnet_v2_m",
        "efficientnet_v2_l",
    ],
    weights: Literal[
        "efficientnet_b0_IMAGENET1K_V1",
        "efficientnet_b1_IMAGENET1K_V1",
        "efficientnet_b1_IMAGENET1K_V2",
        "efficientnet_b2_IMAGENET1K_V1",
        "efficientnet_b3_IMAGENET1K_V1",
        "efficientnet_b4_IMAGENET1K_V1",
        "efficientnet_b5_IMAGENET1K_V1",
        "efficientnet_b6_IMAGENET1K_V1",
        "efficientnet_b7_IMAGENET1K_V1",
        "efficientnet_v2_s_IMAGENET1K_V1",
        "efficientnet_v2_m_IMAGENET1K_V1",
        "efficientnet_v2_l_IMAGENET1K_V1",
    ],
):
    if "_" in weights:
        weights_model = weights.split("_IMAGENET")[0].lower()

        # Check if the specified weights are compatible with the model
        if weights_model != model:
            raise ValueError(
                f"Model {model} is incompatible with weights {weights}. "
                f"Expected weights starting with '{model}'."
            )


def load_efficientnet(
    model: Literal[
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
        "efficientnet_v2_s",
        "efficientnet_v2_m",
        "efficientnet_v2_l",
    ],
    weights: Literal[
        "efficientnet_b0_IMAGENET1K_V1",
        "efficientnet_b1_IMAGENET1K_V1",
        "efficientnet_b1_IMAGENET1K_V2",
        "efficientnet_b2_IMAGENET1K_V1",
        "efficientnet_b3_IMAGENET1K_V1",
        "efficientnet_b4_IMAGENET1K_V1",
        "efficientnet_b5_IMAGENET1K_V1",
        "efficientnet_b6_IMAGENET1K_V1",
        "efficientnet_b7_IMAGENET1K_V1",
        "efficientnet_v2_s_IMAGENET1K_V1",
        "efficientnet_v2_m_IMAGENET1K_V1",
        "efficientnet_v2_l_IMAGENET1K_V1",
    ]
    | None = None,
    n_classes: int = 1000,
    cache: bool = True,
    *,
    key: PRNGKeyArray | None = None,
) -> tuple[EfficientNet, eqx.nn.State]:
    """
    Load an EfficientNet model with optional pre-trained weights.

    Args:
        model: The name of the EfficientNet model to load
        weights: The name of the weights to load (from _MODELS), or None for random init
        n_classes: Number of output classes
        cache: Whether to cache the model
        key: Random key for initialization

    Returns:
        A tuple of (model, state)
    """
    if key is None:
        key = jax.random.key(42)

    match model:
        case "efficientnet_b0":
            efficientnet, state = _efficientnet_b0(key, n_classes)
        case "efficientnet_b1":
            efficientnet, state = _efficientnet_b1(key, n_classes)
        case "efficientnet_b2":
            efficientnet, state = _efficientnet_b2(key, n_classes)
        case "efficientnet_b3":
            efficientnet, state = _efficientnet_b3(key, n_classes)
        case "efficientnet_b4":
            efficientnet, state = _efficientnet_b4(key, n_classes)
        case "efficientnet_b5":
            efficientnet, state = _efficientnet_b5(key, n_classes)
        case "efficientnet_b6":
            efficientnet, state = _efficientnet_b6(key, n_classes)
        case "efficientnet_b7":
            efficientnet, state = _efficientnet_b7(key, n_classes)
        case "efficientnet_v2_s":
            efficientnet, state = _efficientnet_v2_s(key, n_classes)
        case "efficientnet_v2_m":
            efficientnet, state = _efficientnet_v2_m(key, n_classes)
        case "efficientnet_v2_l":
            efficientnet, state = _efficientnet_v2_l(key, n_classes)
        case _:
            raise ValueError(f"Unknown model name: {model}")

    if weights:
        _assert_model_and_weights_fit(model, weights)
        efficientnet, state = _with_weights((efficientnet, state), weights, cache)

    assert efficientnet is not None
    assert state is not None
    return efficientnet, state
