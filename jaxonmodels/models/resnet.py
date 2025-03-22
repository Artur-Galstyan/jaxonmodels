import os
from pathlib import Path
from typing import Type
from urllib.request import urlretrieve

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Literal
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.layers.batch_norm import BatchNorm
from jaxonmodels.statedict2pytree.s2p import (
    convert,
    move_running_fields_to_the_end,
    pytree_to_fields,
    serialize_pytree,
    state_dict_to_fields,
)

_MODELS = {
    "resnet18_IMAGENET1K_V1": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34_IMAGENET1K_V1": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50_IMAGENET1K_V1": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet50_IMAGENET1K_V2": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "resnet101_IMAGENET1K_V1": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet101_IMAGENET1K_V2": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
    "resnet152_IMAGENET1K_V1": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnet152_IMAGENET1K_V2": "https://download.pytorch.org/models/resnet152-f82ba261.pth",
    "resnext50_32X4D_IMAGENET1K_V1": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext50_32X4D_IMAGENET1K_V2": "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
    "resnext101_32X8D_IMAGENET1K_V1": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "resnext101_32X8D_IMAGENET1K_V2": "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
    "resnext101_64X4D_IMAGENET1K_V1": "https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
    "wide_resnet50_2_IMAGENET1K_V1": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet50_2_IMAGENET1K_V2": "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
    "wide_resnet101_2_IMAGENET1K_V1": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
    "wide_resnet101_2_IMAGENET1K_V2": "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
}


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d
    bn: BatchNorm

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        key: PRNGKeyArray,
    ):
        _, subkey = jax.random.split(key)
        self.conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            use_bias=False,
            key=subkey,
        )

        self.bn = BatchNorm(out_channels, axis_name="batch")

    def __call__(
        self,
        x: Float[Array, "c_in h w"],
        state: eqx.nn.State,
        *,
        inference: bool = False,
    ) -> tuple[Float[Array, "c_out*e h/s w/s"], eqx.nn.State]:
        x = self.conv(x)
        x, state = self.bn(x, state, inference=inference)

        return x, state


class BasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: BatchNorm

    downsample: Downsample | None
    expansion: int = eqx.field(static=True, default=1)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        downsample: Downsample | None,
        groups: int,
        base_width: int,
        dilation: int,
        key: PRNGKeyArray,
    ):
        key, *subkeys = jax.random.split(key, 3)

        self.conv1 = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            use_bias=False,
            key=subkeys[0],
        )
        self.bn1 = BatchNorm(size=out_channels, axis_name="batch")

        self.conv2 = eqx.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=subkeys[1],
        )
        self.bn2 = BatchNorm(size=out_channels, axis_name="batch")

        self.downsample = downsample

    def __call__(
        self,
        x: Float[Array, "c h w"],
        state: eqx.nn.State,
        *,
        inference: bool = False,
    ):
        i = x

        x = self.conv1(x)
        x, state = self.bn1(x, state, inference=inference)

        x = jax.nn.relu(x)

        x = self.conv2(x)
        x, state = self.bn2(x, state, inference=inference)

        if self.downsample:
            i, state = self.downsample(i, state, inference=inference)

        x += i
        x = jax.nn.relu(x)

        return x, state


class Bottleneck(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: BatchNorm

    conv3: eqx.nn.Conv2d
    bn3: BatchNorm

    downsample: Downsample | None

    expansion: int = eqx.field(static=True, default=4)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        downsample: Downsample | None,
        groups: int,
        base_width: int,
        dilation: int,
        key: PRNGKeyArray,
    ):
        _, *subkeys = jax.random.split(key, 4)

        width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = eqx.nn.Conv2d(
            in_channels, width, kernel_size=1, use_bias=False, key=subkeys[0]
        )
        self.bn1 = BatchNorm(width, axis_name="batch")

        self.conv2 = eqx.nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            groups=groups,
            dilation=dilation,
            padding=dilation,
            use_bias=False,
            key=subkeys[1],
        )

        self.bn2 = BatchNorm(width, axis_name="batch")

        self.conv3 = eqx.nn.Conv2d(
            width,
            out_channels * self.expansion,
            kernel_size=1,
            key=subkeys[2],
            use_bias=False,
        )

        self.bn3 = BatchNorm(out_channels * self.expansion, axis_name="batch")

        self.downsample = downsample

    def __call__(
        self,
        x: Float[Array, "c_in h w"],
        state: eqx.nn.State,
        *,
        inference: bool = False,
    ) -> tuple[Float[Array, "c_out*e h/s w/s"], eqx.nn.State]:
        i = x

        x = self.conv1(x)
        x, state = self.bn1(x, state, inference=inference)
        x = jax.nn.relu(x)

        x = self.conv2(x)
        x, state = self.bn2(x, state, inference=inference)
        x = jax.nn.relu(x)

        x = self.conv3(x)
        x, state = self.bn3(x, state, inference=inference)

        if self.downsample:
            i, state = self.downsample(i, state, inference=inference)

        x += i
        x = jax.nn.relu(x)
        return x, state


class ResNetBlock(eqx.Module):
    blocks: list[BasicBlock | Bottleneck]

    def __call__(self, x, state, *, inference: bool = False):
        for block in self.blocks:
            x, state = block(x, state, inference=inference)
        return x, state


class ResNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn: BatchNorm
    mp: eqx.nn.MaxPool2d

    layer1: ResNetBlock
    layer2: ResNetBlock
    layer3: ResNetBlock
    layer4: ResNetBlock

    avg: eqx.nn.AdaptiveAvgPool2d
    fc: eqx.nn.Linear

    running_internal_channels: int = eqx.field(static=True, default=64)
    dilation: int = eqx.field(static=True, default=1)

    def __init__(
        self,
        block: Type[BasicBlock | Bottleneck],
        layers: list[int],
        n_classes: int,
        zero_init_residual: bool,
        groups: int,
        width_per_group: int,
        replace_stride_with_dilation: list[bool] | None,
        key: PRNGKeyArray,
        input_channels: int = 3,
    ):
        key, *subkeys = jax.random.split(key, 10)

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"`replace_stride_with_dilation` should either be `None` "
                f"or have a length of 3, got {replace_stride_with_dilation} instead."
            )

        self.conv1 = eqx.nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.running_internal_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=subkeys[0],
        )

        self.bn = BatchNorm(self.running_internal_channels, axis_name="batch")
        self.mp = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResNetBlock(
            self._make_layer(
                block,
                64,
                layers[0],
                stride=1,
                dilate=False,
                groups=groups,
                base_width=width_per_group,
                key=subkeys[1],
            )
        )
        self.layer2 = ResNetBlock(
            self._make_layer(
                block,
                128,
                layers[1],
                stride=2,
                dilate=replace_stride_with_dilation[0],
                groups=groups,
                base_width=width_per_group,
                key=subkeys[2],
            )
        )
        self.layer3 = ResNetBlock(
            self._make_layer(
                block,
                256,
                layers[2],
                stride=2,
                dilate=replace_stride_with_dilation[1],
                groups=groups,
                base_width=width_per_group,
                key=subkeys[3],
            )
        )
        self.layer4 = ResNetBlock(
            self._make_layer(
                block,
                512,
                layers[3],
                stride=2,
                dilate=replace_stride_with_dilation[2],
                groups=groups,
                base_width=width_per_group,
                key=subkeys[4],
            )
        )

        self.avg = eqx.nn.AdaptiveAvgPool2d(target_shape=(1, 1))
        self.fc = eqx.nn.Linear(512 * block.expansion, n_classes, key=subkeys[-1])

        if zero_init_residual:
            # todo: init last bn layer with zero weights
            pass

    def _make_layer(
        self,
        block: Type[BasicBlock | Bottleneck],
        out_channels: int,
        blocks: int,
        stride: int,
        dilate: bool,
        groups: int,
        base_width: int,
        key: PRNGKeyArray,
    ) -> list[BasicBlock | Bottleneck]:
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if (
            stride != 1
            or self.running_internal_channels != out_channels * block.expansion
        ):
            key, subkey = jax.random.split(key)
            downsample = Downsample(
                self.running_internal_channels,
                out_channels * block.expansion,
                stride,
                subkey,
            )
        layers = []

        key, subkey = jax.random.split(key)
        layers.append(
            block(
                in_channels=self.running_internal_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                groups=groups,
                base_width=base_width,
                dilation=previous_dilation,
                key=subkey,
            )
        )

        self.running_internal_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            key, subkey = jax.random.split(key)
            layers.append(
                block(
                    in_channels=self.running_internal_channels,
                    out_channels=out_channels,
                    groups=groups,
                    base_width=base_width,
                    dilation=self.dilation,
                    stride=1,
                    downsample=None,
                    key=subkey,
                )
            )

        return layers

    def __call__(
        self,
        x: Float[Array, "c h w"],
        state: eqx.nn.State,
        *,
        inference: bool = False,
    ) -> tuple[Float[Array, " n_classes"], eqx.nn.State]:
        x = self.conv1(x)
        x, state = self.bn(x, state, inference=inference)
        x = jax.nn.relu(x)
        x = self.mp(x)

        x, state = self.layer1(x, state, inference=inference)
        x, state = self.layer2(x, state, inference=inference)
        x, state = self.layer3(x, state, inference=inference)
        x, state = self.layer4(x, state, inference=inference)

        x = self.avg(x)
        x = jnp.ravel(x)

        x = self.fc(x)

        return x, state


def _kaiming_init(resnet, state, key):
    key, subkey = jax.random.split(key)

    initializer = jax.nn.initializers.he_normal()
    is_conv2d = lambda x: isinstance(x, eqx.nn.Conv2d)
    get_weights = lambda m: [
        x.weight for x in jax.tree.leaves(m, is_leaf=is_conv2d) if is_conv2d(x)
    ]
    weights = get_weights(resnet)
    new_weights = [
        initializer(subkey, weight.shape, jnp.float32)
        for weight, subkey in zip(weights, jax.random.split(subkey, len(weights)))
    ]
    resnet = eqx.tree_at(get_weights, resnet, new_weights)

    return resnet, state


def _with_weights(
    pytree,
    model_name: str,
    cache: bool,
):
    resnet, state = pytree
    if model_name is not None:
        weights_url = _MODELS.get(model_name)
        if weights_url is None:
            raise ValueError(f"No model found for {model_name}")

        # Create directories for caching
        jaxonmodels_dir = os.path.expanduser("~/.jaxonmodels/models")
        os.makedirs(jaxonmodels_dir, exist_ok=True)

        # Check if cached model exists
        if cache:
            if os.path.exists(str(Path(jaxonmodels_dir) / f"{model_name}.eqx")):
                return eqx.tree_deserialise_leaves(
                    str(Path(jaxonmodels_dir) / f"{model_name}.eqx"),
                    (resnet, state),
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
        jaxfields, state_indices = pytree_to_fields((resnet, state))

        resnet, state = convert(
            weights_dict, (resnet, state), jaxfields, state_indices, torchfields
        )

        # Cache the model if requested
        if cache:
            serialize_pytree(
                (resnet, state), str(Path(jaxonmodels_dir) / f"{model_name}.eqx")
            )

    return resnet, state


def _resnet18(key, n_classes=1000):
    """Create a ResNet-18 model without weights"""
    from .resnet import BasicBlock, ResNet  # Import from your module

    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        BasicBlock,
        [2, 2, 2, 2],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )

    resnet, state = _kaiming_init(resnet, state, subkey)

    return resnet, state


# Add all constructors for each ResNet variant
def _resnet34(key, n_classes=1000):
    """Create a ResNet-34 model without weights"""
    from .resnet import BasicBlock, ResNet  # Import from your module

    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        BasicBlock,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )

    resnet, state = _kaiming_init(resnet, state, subkey)

    return resnet, state


def _resnet50(key, n_classes=1000):
    """Create a ResNet-50 model without weights"""

    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )

    resnet, state = _kaiming_init(resnet, state, subkey)

    return resnet, state


def _resnet101(key, n_classes=1000):
    """Create a ResNet-101 model without weights"""

    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )

    resnet, state = _kaiming_init(resnet, state, subkey)

    return resnet, state


def _resnet152(key, n_classes=1000):
    """Create a ResNet-152 model without weights"""

    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 8, 36, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )

    resnet, state = _kaiming_init(resnet, state, subkey)

    return resnet, state


def _resnext50_32x4d(key, n_classes=1000):
    """Create a ResNeXt-50 32x4d model without weights"""

    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=32,
        width_per_group=4,
        replace_stride_with_dilation=None,
        key=key,
    )

    resnet, state = _kaiming_init(resnet, state, subkey)

    return resnet, state


def _resnext101_32x8d(key, n_classes=1000):
    """Create a ResNeXt-101 32x8d model without weights"""

    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=32,
        width_per_group=8,
        replace_stride_with_dilation=None,
        key=key,
    )

    resnet, state = _kaiming_init(resnet, state, subkey)

    return resnet, state


def _resnext101_64x4d(key, n_classes=1000):
    """Create a ResNeXt-101 64x4d model without weights"""

    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=64,
        width_per_group=4,
        replace_stride_with_dilation=None,
        key=key,
    )

    resnet, state = _kaiming_init(resnet, state, subkey)

    return resnet, state


def _wide_resnet50_2(key, n_classes=1000):
    """Create a Wide ResNet-50-2 model without weights"""

    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64 * 2,
        replace_stride_with_dilation=None,
        key=key,
    )

    resnet, state = _kaiming_init(resnet, state, subkey)

    return resnet, state


def _wide_resnet101_2(key, n_classes=1000):
    """Create a Wide ResNet-101-2 model without weights"""

    key, subkey = jax.random.split(key)
    resnet, state = eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64 * 2,
        replace_stride_with_dilation=None,
        key=key,
    )

    resnet, state = _kaiming_init(resnet, state, subkey)

    return resnet, state


# Main function to load models with weights
def load_resnet(
    model: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "resnext101_64x4d",
        "wide_resnet50_2",
        "wide_resnet101_2",
    ],
    weights: str | None = None,
    n_classes: int = 1000,
    cache: bool = True,
    *,
    key: PRNGKeyArray | None = None,
):
    """
    Load a ResNet model with optional pre-trained weights.

    Args:
        model_name: The name of the ResNet model to load
        weights: The name of the weights to load (from _MODELS), or None for random init
        n_classes: Number of output classes
        cache: Whether to cache the model
        key: Random key for initialization

    Returns:
        A tuple of (model, state)
    """
    if key is None:
        key = jax.random.key(42)

    # Create model based on requested architecture
    match model:
        case "resnet18":
            resnet, state = _resnet18(key, n_classes)
        case "resnet34":
            resnet, state = _resnet34(key, n_classes)
        case "resnet50":
            resnet, state = _resnet50(key, n_classes)
        case "resnet101":
            resnet, state = _resnet101(key, n_classes)
        case "resnet152":
            resnet, state = _resnet152(key, n_classes)
        case "resnext50_32x4d":
            resnet, state = _resnext50_32x4d(key, n_classes)
        case "resnext101_32x8d":
            resnet, state = _resnext101_32x8d(key, n_classes)
        case "resnext101_64x4d":
            resnet, state = _resnext101_64x4d(key, n_classes)
        case "wide_resnet50_2":
            resnet, state = _wide_resnet50_2(key, n_classes)
        case "wide_resnet101_2":
            resnet, state = _wide_resnet101_2(key, n_classes)
        case _:
            raise ValueError(f"Unknown model name: {model}")

    # Load weights if requested
    if weights is not None:
        resnet, state = _with_weights((resnet, state), weights, cache)

    assert resnet is not None
    assert state is not None
    return resnet, state
