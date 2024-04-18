import functools as ft
from collections.abc import Callable

import equinox as eqx
import jax
from beartype.typing import Optional, Union
from jaxtyping import Array, PRNGKeyArray, PyTree


def make_resnet_layer(
    resnet: "ResNet",
    block: Union["BasicBlock", "Bottleneck"],
    planes: int,
    blocks: int,
    stride: int = 1,
    dilate: bool = False,
) -> "ResNetLayer":
    # TODO
    return ResNetLayer()


class ResNetLayer(eqx.Module):
    pass


class BasicBlock(eqx.Module):
    expansion: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm

    downsample: Optional[PyTree]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[PyTree] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., PyTree]] = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.expansion = 1
        if norm_layer is None:
            norm_layer = eqx.nn.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        key, conv1_key = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            use_bias=False,
            dilation=dilation,
            key=conv1_key,
        )
        self.bn1 = norm_layer(planes, axis_name="batch")
        self.conv2 = eqx.nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            use_bias=False,
            dilation=dilation,
            key=conv1_key,
        )
        self.bn2 = norm_layer(planes, axis_name="batch")
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        identity = x

        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = jax.nn.relu(out)

        return out, state


class Bottleneck(eqx.Module):
    pass


class ResNet(eqx.Module):
    inplanes: int = eqx.field(static=True)
    dilation: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    base_width: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    maxpool: eqx.nn.MaxPool2d
    avgpool: eqx.nn.AdaptiveAvgPool2d
    fc: eqx.nn.Linear

    def __init__(
        self,
        block: Union[BasicBlock, Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., eqx.Module]] = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        strides = [2] * 3
        key, conv1_key, fc_key = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=key,
        )

        norm_layer = (
            ft.partial(eqx.nn.BatchNorm, axis_name="batch")
            if norm_layer is None
            else norm_layer
        )
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = eqx.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = eqx.nn.Linear(512 * block.expansion, num_classes, key=fc_key)
