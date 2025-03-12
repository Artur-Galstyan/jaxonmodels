from typing import Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt

from jaxonmodels.layers.batch_norm import BatchNorm


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d
    bn: BatchNorm

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        key: jt.PRNGKeyArray,
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
        self, x: jt.Float[jt.Array, "c_in h w"], state: eqx.nn.State
    ) -> tuple[jt.Float[jt.Array, "c_out*e h/s w/s"], eqx.nn.State]:
        x = self.conv(x)
        x, state = self.bn(x, state)

        return x, state


class BasicBlock(eqx.Module):
    downsample: Downsample | None

    conv1: eqx.nn.Conv2d
    bn1: BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: BatchNorm

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
        key: jt.PRNGKeyArray,
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

    def __call__(self, x: jt.Float[jt.Array, "c h w"], state: eqx.nn.State):
        i = x

        x = self.conv1(x)
        x, state = self.bn1(x, state)

        x = jax.nn.relu(x)

        x = self.conv2(x)
        x, state = self.bn2(x, state)

        if self.downsample:
            i, state = self.downsample(i, state)

        x += i
        x = jax.nn.relu(x)

        return x, state


class Bottleneck(eqx.Module):
    downsample: Downsample | None

    conv1: eqx.nn.Conv2d
    bn1: BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: BatchNorm

    conv3: eqx.nn.Conv2d
    bn3: BatchNorm

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
        key: jt.PRNGKeyArray,
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
        self, x: jt.Float[jt.Array, "c_in h w"], state: eqx.nn.State
    ) -> tuple[jt.Float[jt.Array, "c_out*e h/s w/s"], eqx.nn.State]:
        i = x

        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)

        x = self.conv2(x)
        x, state = self.bn2(x, state)
        x = jax.nn.relu(x)

        x = self.conv3(x)
        x, state = self.bn3(x, state)

        if self.downsample:
            i, state = self.downsample(i, state)

        x += i
        x = jax.nn.relu(x)
        return x, state


class ResNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn: BatchNorm
    mp: eqx.nn.MaxPool2d

    layer1: list[BasicBlock | Bottleneck]
    layer2: list[BasicBlock | Bottleneck]
    layer3: list[BasicBlock | Bottleneck]
    layer4: list[BasicBlock | Bottleneck]

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
        key: jt.PRNGKeyArray,
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

        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            stride=1,
            dilate=False,
            groups=groups,
            base_width=width_per_group,
            key=subkeys[1],
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            groups=groups,
            base_width=width_per_group,
            key=subkeys[2],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            groups=groups,
            base_width=width_per_group,
            key=subkeys[3],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            groups=groups,
            base_width=width_per_group,
            key=subkeys[4],
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
        key: jt.PRNGKeyArray,
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
        self, x: jt.Float[jt.Array, "c h w"], state: eqx.nn.State
    ) -> tuple[jt.Float[jt.Array, " n_classes"], eqx.nn.State]:
        x = self.conv1(x)
        x, state = self.bn(x, state)
        x = jax.nn.relu(x)
        x = self.mp(x)

        for layer in self.layer1:
            x, state = layer(x, state)

        for layer in self.layer2:
            x, state = layer(x, state)

        for layer in self.layer3:
            x, state = layer(x, state)

        for layer in self.layer4:
            x, state = layer(x, state)

        x = self.avg(x)
        x = jnp.ravel(x)

        x = self.fc(x)

        return x, state


def resnet18(key: jt.PRNGKeyArray, n_classes=1000) -> tuple[ResNet, eqx.nn.State]:
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

    # initializer = jax.nn.initializers.he_normal()
    # is_conv2d = lambda x: isinstance(x, eqx.nn.Conv2d)
    # get_weights = lambda m: [
    #     x.weight for x in jax.tree.leaves(m, is_leaf=is_conv2d) if is_conv2d(x)
    # ]
    # weights = get_weights(resnet)
    # new_weights = [
    #     initializer(subkey, weight.shape, jnp.float32)
    #     for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    # ]
    # resnet = eqx.tree_at(get_weights, resnet, new_weights)

    return resnet, state


def resnet34(key: jt.PRNGKeyArray, n_classes=1000) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        BasicBlock,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnet50(key: jt.PRNGKeyArray, n_classes=1000) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnet101(key: jt.PRNGKeyArray, n_classes=1000) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnet152(key: jt.PRNGKeyArray, n_classes=1000) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 8, 36, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnext50_32x4d(
    key: jt.PRNGKeyArray, n_classes=1000
) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=32,
        width_per_group=4,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnext101_32x8d(
    key: jt.PRNGKeyArray, n_classes=1000
) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=32,
        width_per_group=8,
        replace_stride_with_dilation=None,
        key=key,
    )


def resnext101_64x4d(
    key: jt.PRNGKeyArray, n_classes=1000
) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=64,
        width_per_group=4,
        replace_stride_with_dilation=None,
        key=key,
    )


def wide_resnet50_2(
    key: jt.PRNGKeyArray, n_classes=1000
) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 6, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64 * 2,
        replace_stride_with_dilation=None,
        key=key,
    )


def wide_resnet101_2(
    key: jt.PRNGKeyArray, n_classes=1000
) -> tuple[ResNet, eqx.nn.State]:
    return eqx.nn.make_with_state(ResNet)(
        Bottleneck,
        [3, 4, 23, 3],
        n_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64 * 2,
        replace_stride_with_dilation=None,
        key=key,
    )
