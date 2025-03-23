import copy
import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Callable, Sequence
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions.functions import stochastic_depth
from jaxonmodels.layers.batch_norm import BatchNorm


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
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
        return _make_divisible(channels * width_mult, 8, min_value)


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
    norm_layer: eqx.Module | None
    activation: eqx.nn.Lambda | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = (1, 1),
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] = (0, 0),
        dilation: int | Sequence[int] = (1, 1),
        groups: int = 1,
        use_bias: bool | None = None,
        padding_mode: str = "ZEROS",
        dtype=None,
        norm_layer: Callable[..., eqx.Module] | None = BatchNorm,
        activation_layer: Callable[..., Array] | None = jax.nn.relu,
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
                    else len(dilation)
                )
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple(
                    (kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim)
                )
        if use_bias is None:
            use_bias = norm_layer is None

        self.layers = []
        key, subkey = jax.random.split(key)

        self.conv2d = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            key=subkey,
        )

        if norm_layer is not None:
            self.norm = norm_layer(out_channels, axis_name=axis_name)

        if activation_layer is not None:
            self.activation = eqx.nn.Lambda(activation_layer)

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


class SqueezeExcitation(eqx.Module):
    avgpool: eqx.nn.AdaptiveAvgPool2d
    fc1: eqx.nn.Conv2d
    fc2: eqx.nn.Conv2d

    def __init__(
        self, input_channels: int, squeeze_channels: int, *, key: PRNGKeyArray
    ) -> None:
        self.avgpool = eqx.nn.AdaptiveAvgPool2d(1)
        key, subkey = jax.random.split(key)
        self.fc1 = eqx.nn.Conv2d(input_channels, squeeze_channels, 1, key=key)
        self.fc2 = eqx.nn.Conv2d(squeeze_channels, input_channels, 1, key=subkey)

    def _scale(
        self,
        input: Array,
        activation: Callable[..., Array] = jax.nn.relu,
        scale_activation: Callable[..., Array] = jax.nn.sigmoid,
    ) -> Array:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = activation(scale)
        scale = self.fc2(scale)
        return scale_activation(scale)

    def __call__(
        self,
        input: Array,
        activation: Callable[..., Array] = jax.nn.relu,
        scale_activation: Callable[..., Array] = jax.nn.sigmoid,
    ) -> Array:
        scale = self._scale(input)
        return scale * input


class StochasticDepth(eqx.Module):
    p: float = eqx.field(static=True)
    mode: str = eqx.field(static=True)

    def __init__(self, p: float, mode: str) -> None:
        self.p = p
        self.mode = mode

    def __call__(self, input: Array, inference: bool, key: PRNGKeyArray) -> Array:
        return stochastic_depth(input, self.p, self.mode, inference, key)


class MBConv(eqx.Module):
    use_res_connect: bool = eqx.field(static=True)

    expand_conv2d: Conv2dNormActivation | None
    depthwise_conv2d: Conv2dNormActivation
    se_layer: SqueezeExcitation

    project_conv2d: Conv2dNormActivation
    stochastic_depth: StochasticDepth

    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        *,
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
        result = self.se_layer(result)

        if self.use_res_connect:
            result = self.stochastic_depth(result, inference=inference, key=key)
            result += x
        return result


class FusedMBConv(eqx.Module):
    fused_expand: Conv2dNormActivation
    project: Conv2dNormActivation | None
    stochastic_depth: StochasticDepth

    use_res_connect: bool = eqx.field(static=True)

    def __init__(
        self,
        cnf: FusedMBConvConfig,
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

    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
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
                stage.append(block_cnf.block(block_cnf, sd_prob, BatchNorm, key=subkey))
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
            dropout, lastconv_output_channels, num_classes, subkey
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
