import math
from dataclasses import dataclass

import equinox as eqx
import jax
from beartype.typing import Callable, Sequence
from jaxtyping import Array, PRNGKeyArray


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


class MBConv(eqx.Module):
    pass


class FusedMBConv(eqx.Module):
    pass


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

    def forward(
        self,
        input: Array,
        activation: Callable[..., Array] = jax.nn.relu,
        scale_activation: Callable[..., Array] = jax.nn.sigmoid,
    ) -> Array:
        scale = self._scale(input)
        return scale * input


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
