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
from jaxtyping import Array, PRNGKeyArray

from jaxonmodels.functions import (
    default_floating_dtype,
    make_divisible,
)
from jaxonmodels.functions.utils import dtype_to_str
from jaxonmodels.layers import (
    BatchNorm,
    ConvNormActivation,
    SqueezeExcitation,
    StochasticDepth,
)
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


class MBConv(eqx.Module):
    use_res_connect: bool = eqx.field(static=True)

    expand_conv2d: ConvNormActivation | None
    depthwise_conv2d: ConvNormActivation
    se_layer: SqueezeExcitation

    project_conv2d: ConvNormActivation
    stochastic_depth: StochasticDepth

    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        key: PRNGKeyArray,
        dtype: Any,
    ):
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        activation_layer = jax.nn.silu

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        self.expand_conv2d = None
        if expanded_channels != cnf.input_channels:
            key, subkey = jax.random.split(key)
            self.expand_conv2d = ConvNormActivation(
                2,
                cnf.input_channels,
                expanded_channels,
                kernel_size=1,
                norm_layer=BatchNorm,
                activation_layer=activation_layer,
                key=subkey,
                dtype=dtype,
            )

        # depthwise
        key, subkey = jax.random.split(key)
        self.depthwise_conv2d = ConvNormActivation(
            2,
            expanded_channels,
            expanded_channels,
            kernel_size=cnf.kernel,
            stride=cnf.stride,
            groups=expanded_channels,
            norm_layer=BatchNorm,
            activation_layer=activation_layer,
            key=subkey,
            dtype=dtype,
        )
        squeeze_channels = max(1, cnf.input_channels // 4)

        key, subkey = jax.random.split(key)
        self.se_layer = SqueezeExcitation(
            expanded_channels, squeeze_channels, key=subkey, dtype=dtype
        )

        # project
        key, subkey = jax.random.split(key)
        self.project_conv2d = ConvNormActivation(
            2,
            expanded_channels,
            cnf.out_channels,
            kernel_size=1,
            norm_layer=BatchNorm,
            activation_layer=None,
            key=subkey,
            dtype=dtype,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def __call__(
        self, x: Array, state: eqx.nn.State, inference: bool, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
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
    fused_expand: ConvNormActivation
    project: ConvNormActivation | None
    stochastic_depth: StochasticDepth

    use_res_connect: bool = eqx.field(static=True)

    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        key: PRNGKeyArray,
        dtype: Any,
    ):
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        self.project = None
        if expanded_channels != cnf.input_channels:
            # fused expand
            key, subkey, subkey2 = jax.random.split(key, 3)
            self.fused_expand = ConvNormActivation(
                2,
                cnf.input_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                norm_layer=BatchNorm,
                activation_layer=jax.nn.silu,
                key=subkey,
                dtype=dtype,
            )

            # project
            self.project = ConvNormActivation(
                2,
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=BatchNorm,
                activation_layer=None,
                key=subkey2,
                dtype=dtype,
            )
        else:
            key, subkey = jax.random.split(key)
            self.fused_expand = ConvNormActivation(
                2,
                cnf.input_channels,
                cnf.out_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                norm_layer=BatchNorm,
                activation_layer=jax.nn.silu,
                key=subkey,
                dtype=dtype,
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
        keys = jax.random.split(key, len(self.layers) + 1)
        for i, stage in enumerate(self.layers):
            x, state = stage(x, state, inference, keys[i])

        return x, state


class Classifier(eqx.Module):
    dropout: eqx.nn.Dropout
    linear: eqx.nn.Linear

    def __init__(
        self,
        p: float,
        in_features: int,
        out_features: int,
        key: PRNGKeyArray,
        dtype: Any,
    ):
        self.dropout = eqx.nn.Dropout(p=p)

        self.linear = eqx.nn.Linear(in_features, out_features, key=key, dtype=dtype)

    def __call__(self, x: Array, inference: bool, key: PRNGKeyArray) -> Array:
        x = self.dropout(x, inference=inference, key=key)
        x = self.linear(x)
        return x


class EfficientNet(eqx.Module):
    features: list[ConvNormActivation | MBConv | FusedMBConv | InvertedResidualBlock]
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
        dtype: Any | None = None,  # Added optional dtype parameter
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
                " or List[FusedMBConvConfig]"
            )

        key, *subkeys = jax.random.split(key, 10)

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.features = []
        self.features.append(
            ConvNormActivation(
                2,
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=BatchNorm,
                activation_layer=jax.nn.silu,
                key=subkeys[0],
                dtype=dtype,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: list[MBConv | FusedMBConv] = []
            num_layers = cnf.num_layers
            stage_keys = jax.random.split(key, num_layers + 1)
            key = stage_keys[0]
            for i in range(num_layers):
                block_cnf = copy.copy(cnf)

                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )
                stage.append(
                    block_cnf.block(block_cnf, sd_prob, stage_keys[i], dtype=dtype)
                )
                stage_block_id += 1

            self.features.append(InvertedResidualBlock(stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.features.append(
            ConvNormActivation(
                2,
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=BatchNorm,
                activation_layer=jax.nn.silu,
                key=subkey1,
                dtype=dtype,
            )
        )

        self.avgpool = eqx.nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier(
            dropout,
            lastconv_output_channels,
            n_classes,
            subkey2,
            dtype=dtype,
        )
        # TODO: kaiming init

    def __call__(
        self, x: Array, state: eqx.nn.State, inference: bool, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        num_feature_layers = len(self.features)
        keys = jax.random.split(key, num_feature_layers + 2)
        feature_keys = keys[:num_feature_layers]
        classifier_key = keys[-1]

        for i, feature in enumerate(self.features):
            x, state = feature(x, state, inference, feature_keys[i])

        x = self.avgpool(x)
        x = jnp.ravel(x)

        x = self.classifier(x, inference, classifier_key)

        return x, state


def _efficientnet(
    inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig],
    dropout: float,
    last_channel: int | None,
    n_classes: int,
    key: PRNGKeyArray,
    dtype: Any,
) -> tuple[EfficientNet, eqx.nn.State]:
    model, state = eqx.nn.make_with_state(EfficientNet)(
        inverted_residual_setting,
        dropout,
        last_channel=last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
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
    dtype: Any,
):
    """Load pre-trained weights for the model"""
    efficientnet, state = pytree
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
                (efficientnet, state),
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
            # EfficientNet doesn't have logit_scale like CLIP
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
    jaxfields, state_indices = pytree_to_fields((efficientnet, state))

    efficientnet, state = convert(
        weights_dict,
        (efficientnet, state),
        jaxfields,
        state_indices,
        torchfields,
        dtype=dtype,
    )

    if cache:
        serialize_pytree((efficientnet, state), cache_filepath)

    return efficientnet, state


def _efficientnet_b0(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b0", width_mult=1.0, depth_mult=1.0
    )
    return _efficientnet(
        inverted_residual_setting,
        0.2,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
    )


def _efficientnet_b1(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b1", width_mult=1.0, depth_mult=1.1
    )
    return _efficientnet(
        inverted_residual_setting,
        0.2,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
    )


def _efficientnet_b2(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b2", width_mult=1.1, depth_mult=1.2
    )
    return _efficientnet(
        inverted_residual_setting,
        0.3,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
    )


def _efficientnet_b3(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b3", width_mult=1.2, depth_mult=1.4
    )
    return _efficientnet(
        inverted_residual_setting,
        0.3,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
    )


def _efficientnet_b4(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b4", width_mult=1.4, depth_mult=1.8
    )
    return _efficientnet(
        inverted_residual_setting,
        0.4,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
    )


def _efficientnet_b5(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b5", width_mult=1.6, depth_mult=2.2
    )
    return _efficientnet(
        inverted_residual_setting,
        0.4,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
    )


def _efficientnet_b6(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b6", width_mult=1.8, depth_mult=2.6
    )
    return _efficientnet(
        inverted_residual_setting,
        0.5,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
    )


def _efficientnet_b7(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b7", width_mult=2.0, depth_mult=3.1
    )
    return _efficientnet(
        inverted_residual_setting,
        0.5,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
    )


def _efficientnet_v2_s(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        0.2,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
    )


def _efficientnet_v2_m(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return _efficientnet(
        inverted_residual_setting,
        0.3,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
    )


def _efficientnet_v2_l(key, n_classes=1000, dtype: Any = None):
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return _efficientnet(
        inverted_residual_setting,
        0.4,
        last_channel,
        n_classes=n_classes,
        key=key,
        dtype=dtype,
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
    dtype: Any | None = None,  # Added optional dtype parameter
) -> tuple[EfficientNet, eqx.nn.State]:
    """
    Load an EfficientNet model with optional pre-trained weights.

    Args:
        model: The name of the EfficientNet model to load
        weights: The name of the weights to load (from _MODELS), or None for random init
        n_classes: Number of output classes
        cache: Whether to cache the model and weights.
        key: Random key for initialization. Defaults to jax.random.key(42).
        dtype: The data type for the model's parameters (e.g., jnp.float32).
               Defaults to jaxonmodels.functions.default_floating_dtype().

    Returns:
        A tuple of (model, state)
    """
    if key is None:
        key = jax.random.key(42)

    # Determine default dtype if not provided before passing to constructors
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None  # Ensure dtype is set

    efficientnet, state = None, None  # Initialize

    match model:
        case "efficientnet_b0":
            efficientnet, state = _efficientnet_b0(key, n_classes, dtype=dtype)
        case "efficientnet_b1":
            efficientnet, state = _efficientnet_b1(key, n_classes, dtype=dtype)
        case "efficientnet_b2":
            efficientnet, state = _efficientnet_b2(key, n_classes, dtype=dtype)
        case "efficientnet_b3":
            efficientnet, state = _efficientnet_b3(key, n_classes, dtype=dtype)
        case "efficientnet_b4":
            efficientnet, state = _efficientnet_b4(key, n_classes, dtype=dtype)
        case "efficientnet_b5":
            efficientnet, state = _efficientnet_b5(key, n_classes, dtype=dtype)
        case "efficientnet_b6":
            efficientnet, state = _efficientnet_b6(key, n_classes, dtype=dtype)
        case "efficientnet_b7":
            efficientnet, state = _efficientnet_b7(key, n_classes, dtype=dtype)
        case "efficientnet_v2_s":
            efficientnet, state = _efficientnet_v2_s(key, n_classes, dtype=dtype)
        case "efficientnet_v2_m":
            efficientnet, state = _efficientnet_v2_m(key, n_classes, dtype=dtype)
        case "efficientnet_v2_l":
            efficientnet, state = _efficientnet_v2_l(key, n_classes, dtype=dtype)
        case _:
            raise ValueError(f"Unknown model name: {model}")

    if weights:
        _assert_model_and_weights_fit(model, weights)
        efficientnet, state = _with_weights(
            (efficientnet, state), weights, cache, dtype=dtype
        )

    assert efficientnet is not None
    assert state is not None
    return efficientnet, state
