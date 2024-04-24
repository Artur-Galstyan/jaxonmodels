import functools as ft
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Optional, Type, Union
from equinox.nn import State
from jaxtyping import Array, PRNGKeyArray, PyTree

from jaxonmodels.commons.utils import (
    get_node,
    pytorch_state_dict_str_to_pytree_str,
)


def _iterate_layers(layer: list[PyTree], x: Array, state: State):
    for l in layer:
        if hasattr(l, "is_stateful") and l.is_stateful():
            x, state = l(x, state)
        else:
            x = l(x)
    return x, state


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


class BasicBlock(eqx.Module):
    stride: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm

    downsample: Optional[list[PyTree]]

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
        if norm_layer is None:
            norm_layer = eqx.nn.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        key, conv1_key, conv2_key = jax.random.split(key, 3)
        self.conv1 = conv3x3(inplanes, planes, stride, key=conv1_key)
        self.bn1 = norm_layer(planes, axis_name="batch")
        self.conv2 = conv3x3(planes, planes, key=conv2_key)
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
            identity, state = _iterate_layers(self.downsample, x, state)

        out += identity
        out = jax.nn.relu(out)

        return out, state

    def is_stateful(self):
        return True


class Bottleneck(eqx.Module):
    stride: int = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm
    bn3: eqx.nn.BatchNorm

    downsample: Optional[list[PyTree]]

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
            out = self.relu(out)

            out = self.conv2(out)
            out, state = self.bn2(out, state)
            out = jax.nn.relu(out)

            out = self.conv3(out)
            out, state = self.bn3(out, state)

            if self.downsample is not None:
                identity, state = _iterate_layers(self.downsample, x, state)

            out += identity
            out = jax.nn.relu(out)

            return out, state

    def is_stateful(self):
        return True


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

    layer1: list[PyTree]
    layer2: list[PyTree]
    layer3: list[PyTree]
    layer4: list[PyTree]

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[list[bool]] = None,
        norm_layer: Optional[Callable[..., PyTree]] = None,
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

        norm_layer = (
            ft.partial(eqx.nn.BatchNorm, axis_name="batch")
            if norm_layer is None
            else norm_layer
        )
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_resnet_layer(
            self, block, self.inplanes, 4, norm_layer=norm_layer, key=layer1_key
        )

        self.layer2 = _make_resnet_layer(
            self,
            block,
            128,
            layers[1],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[0],
            key=layer2_key,
        )
        self.layer3 = _make_resnet_layer(
            self,
            block,
            256,
            layers[2],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[1],
            key=layer3_key,
        )
        self.layer4 = _make_resnet_layer(
            self,
            block,
            512,
            layers[3],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[2],
            key=layer4_key,
        )
        self.avgpool = eqx.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = eqx.nn.Linear(512 * _get_expansion(block), num_classes, key=fc_key)

    def __call__(self, x: Array, state: State) -> tuple[Array, State]:
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        x = self.maxpool(x)

        x, state = _iterate_layers(self.layer1, x, state)
        x, state = _iterate_layers(self.layer2, x, state)
        x, state = _iterate_layers(self.layer3, x, state)
        x, state = _iterate_layers(self.layer4, x, state)

        x = self.avgpool(x)
        x = x.reshape(-1)
        x = self.fc(x)

        return x, state


def _make_resnet_layer(
    resnet: ResNet,
    block: Type[Union[BasicBlock, Bottleneck]],
    planes: int,
    blocks: int,
    norm_layer: Callable[..., PyTree],
    stride: int = 1,
    dilate: bool = False,
    *,
    key: PRNGKeyArray,
) -> list[PyTree]:
    downsample = None
    previous_dilation = resnet.dilation
    if dilate:
        resnet.dilation *= stride
        stride = 1
    key, downsample_key = jax.random.split(key)
    if stride != 1 or resnet.inplanes != planes * _get_expansion(block):
        downsample = [
            conv1x1(
                resnet.inplanes,
                planes * _get_expansion(block),
                stride,
                key=downsample_key,
            ),
            norm_layer(planes * _get_expansion(block)),
        ]
    key, *layer_keys = jax.random.split(key, blocks + 1)
    layers = []
    layers.append(
        block(
            resnet.inplanes,
            planes,
            stride,
            downsample,
            resnet.groups,
            resnet.base_width,
            previous_dilation,
            norm_layer,
            key=layer_keys[0],
        )
    )
    resnet.inplanes = planes * _get_expansion(block)
    for i in range(1, blocks):
        layers.append(
            block(
                resnet.inplanes,
                planes,
                groups=resnet.groups,
                base_width=resnet.base_width,
                dilation=resnet.dilation,
                norm_layer=norm_layer,
                key=layer_keys[i],
            )
        )

    return layers


def _get_expansion(
    block: Type[Union[BasicBlock, Bottleneck]],
) -> int:
    return 1 if block is BasicBlock else 4


def resnet18(
    key: Optional[PRNGKeyArray] = None,
    load_weights: bool = False,
    target_dir: Optional[str] = None,
) -> tuple[ResNet, State]:
    key = jax.random.PRNGKey(22) if key is None else key
    resnet, state = eqx.nn.make_with_state(ResNet)(BasicBlock, [2, 2, 2, 2], key=key)
    if load_weights:
        from torchvision.models import resnet18, ResNet18_Weights

        state_dict = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()

        for k in state_dict:
            pytree_string = pytorch_state_dict_str_to_pytree_str(k)
            where = ft.partial(get_node, targets=pytree_string.split(".")[1:])
            if where(resnet) is not None:
                resnet = eqx.tree_at(where, resnet, jnp.array(state_dict[k].numpy()))
    return resnet, state


def resnet50(key: Optional[PRNGKeyArray] = None) -> tuple[ResNet, State]:
    key = jax.random.PRNGKey(22) if key is None else key
    return eqx.nn.make_with_state(ResNet)(Bottleneck, [3, 4, 6, 3], key=key)
