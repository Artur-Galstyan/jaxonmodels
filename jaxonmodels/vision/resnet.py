from collections.abc import Callable

import equinox as eqx
import jax
from beartype.typing import Optional, Type, Union
from equinox.nn import BatchNorm, State
from jaxtyping import Array, PRNGKeyArray, PyTree


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    *,
    key: PRNGKeyArray,
) -> eqx.nn.Conv2d:
    return eqx.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        use_bias=False,
        dilation=dilation,
        key=key,
    )


def conv1x1(
    in_planes: int, out_planes: int, stride: int = 1, *, key: PRNGKeyArray
) -> eqx.nn.Conv2d:
    return eqx.nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, use_bias=False, key=key
    )


class DownsampleBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.BatchNorm

    def __init__(
        self, in_planes: int, out_planes: int, stride: int, *, key: PRNGKeyArray
    ):
        self.conv = conv1x1(in_planes, out_planes, stride, key=key)
        self.norm = BatchNorm(out_planes, axis_name="batch")

    def __call__(self, x: Array, state: State) -> tuple[Array, State]:
        x = self.conv(x)
        x, state = self.norm(x, state)
        return x, state


class BasicBlock(eqx.Module):
    stride: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm

    downsample: Optional[DownsampleBlock]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[DownsampleBlock] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        *,
        key: PRNGKeyArray,
    ) -> None:
        # print(
        #     f"{inplanes=}",
        #     f"{planes=}",
        #     f"{stride=}",
        #     f"{downsample=}",
        #     f"{groups=}",
        #     f"{base_width=}",
        #     f"{dilation=}",
        # )
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        key, conv1_key, conv2_key = jax.random.split(key, 3)
        self.conv1 = conv3x3(inplanes, planes, stride, key=conv1_key)
        self.bn1 = eqx.nn.BatchNorm(planes, axis_name="batch")
        self.conv2 = conv3x3(planes, planes, key=conv2_key)
        self.bn2 = eqx.nn.BatchNorm(planes, axis_name="batch")
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
            identity, state = self.downsample(x, state)

        out += identity
        out = jax.nn.relu(out)

        return out, state


class Bottleneck(eqx.Module):
    stride: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm
    bn3: eqx.nn.BatchNorm

    downsample: Optional[DownsampleBlock]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[DownsampleBlock] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., PyTree]] = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        expansion = 4
        norm_layer = eqx.nn.BatchNorm if norm_layer is None else norm_layer
        width = int(planes * (base_width / 64.0)) * groups
        key, *subkeys = jax.random.split(key, 4)
        self.conv1 = conv1x1(inplanes, width, key=subkeys[0])
        self.bn1 = norm_layer(width, axis_name="batch")
        self.conv2 = conv3x3(width, width, stride, groups, dilation, key=subkeys[1])
        self.bn2 = norm_layer(width, axis_name="batch")
        self.conv3 = conv1x1(width, planes * expansion, key=subkeys[2])
        self.bn3 = norm_layer(planes * expansion, axis_name="batch")
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: Array, state: State) -> tuple[Array, State]:
        identity = x

        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)
        out = jax.nn.relu(out)

        out = self.conv3(out)
        out, state = self.bn3(out, state)

        if self.downsample is not None:
            identity, state = self.downsample(x, state)

        out += identity
        out = jax.nn.relu(out)

        return out, state


class ResNetLayer(eqx.Module):
    blocks: list[BasicBlock | Bottleneck]

    def __init__(
        self,
        block: Type[BasicBlock | Bottleneck],
        blocks: int,
        in_planes: int,
        out_planes: int,
        groups: int,
        base_width: int,
        stride: int,
        dilation: int,
        *,
        key: PRNGKeyArray,
    ):
        key, downsample_key = jax.random.split(key)
        if stride != 1 or in_planes != out_planes * _get_expansion(block):
            downsample = DownsampleBlock(
                in_planes,
                out_planes * _get_expansion(block),
                stride,
                key=downsample_key,
            )
        else:
            downsample = None

        key, *layer_keys = jax.random.split(key, blocks + 1)
        layers = []
        layers.append(
            block(
                in_planes,
                out_planes,
                stride,
                downsample,
                groups,
                base_width,
                dilation,
                key=layer_keys[0],
            )
        )
        for i in range(1, blocks):
            layers.append(
                block(
                    out_planes * _get_expansion(block),
                    out_planes,
                    groups=groups,
                    base_width=base_width,
                    dilation=dilation,
                    key=layer_keys[i],
                )
            )
        self.blocks = layers

    def __call__(self, x: Array, state: State) -> tuple[Array, State]:
        for block in self.blocks:
            x, state = block(x, state)
        return x, state


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

    layer1: ResNetLayer
    layer2: ResNetLayer
    layer3: ResNetLayer
    layer4: ResNetLayer

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] = [False, False, False],
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        key, conv1_key, fc_key, layer1_key, layer2_key, layer3_key, layer4_key = (
            jax.random.split(key, 7)
        )
        self.conv1 = eqx.nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=key,
        )

        self.bn1 = eqx.nn.BatchNorm(self.inplanes, axis_name="batch")
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        planes = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        for i in range(1, 4):
            if replace_stride_with_dilation[i - 1]:
                strides[i] = 1
        in_planes = [64, 64, 128, 256]
        in_planes = [i * _get_expansion(block) for i in in_planes]
        dilations = []
        for stride, dilate in zip(strides, replace_stride_with_dilation + [False]):
            if dilate:
                dilations.append(self.dilation * stride)
            else:
                dilations.append(self.dilation)

        self.layer1 = ResNetLayer(
            block,
            layers[0],
            in_planes[0],
            planes[0],
            groups,
            width_per_group,
            strides[0],
            dilations[0],
            key=layer1_key,
        )

        self.layer2 = ResNetLayer(
            block,
            layers[1],
            in_planes[1],
            planes[1],
            groups,
            width_per_group,
            strides[1],
            dilations[1],
            key=layer2_key,
        )

        self.layer3 = ResNetLayer(
            block,
            layers[2],
            in_planes[2],
            planes[2],
            groups,
            width_per_group,
            strides[2],
            dilations[2],
            key=layer3_key,
        )

        self.layer4 = ResNetLayer(
            block,
            layers[3],
            in_planes[3],
            planes[3],
            groups,
            width_per_group,
            strides[3],
            dilations[3],
            key=layer4_key,
        )

        self.avgpool = eqx.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = eqx.nn.Linear(512 * _get_expansion(block), num_classes, key=fc_key)

    def __call__(self, x: Array, state: State) -> tuple[Array, State]:
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        x = self.maxpool(x)

        x, state = self.layer1(x, state)
        x, state = self.layer2(x, state)
        x, state = self.layer3(x, state)
        x, state = self.layer4(x, state)

        x = self.avgpool(x)
        x = x.reshape(-1)
        x = self.fc(x)

        return x, state


def _get_expansion(
    block: Type[Union[BasicBlock, Bottleneck]],
) -> int:
    return 1 if block is BasicBlock else 4


def resnet18(
    key: Optional[PRNGKeyArray] = None,
) -> tuple[ResNet, State]:
    key = jax.random.PRNGKey(22) if key is None else key
    resnet, state = eqx.nn.make_with_state(ResNet)(BasicBlock, [2, 2, 2, 2], key=key)
    return resnet, state


def resnet50(key: Optional[PRNGKeyArray] = None) -> tuple[ResNet, State]:
    key = jax.random.PRNGKey(22) if key is None else key
    return eqx.nn.make_with_state(ResNet)(Bottleneck, [3, 4, 6, 3], key=key)
