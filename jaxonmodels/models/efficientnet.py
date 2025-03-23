import math
from dataclasses import dataclass

import equinox as eqx
import jax
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
    block: Callable[..., eqx.Module]

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
        block: Callable[..., eqx.Module] | None = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
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
        block: Callable[..., eqx.Module] | None = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
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
        self, x: Float[Array, "c h w"], state: eqx.nn.State, inference: bool = False
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
    pass


class EfficientNet(eqx.Module):
    pass

    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig | FusedMBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        last_channel: int | None = None,
    ):
        pass
