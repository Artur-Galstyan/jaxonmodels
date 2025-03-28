import equinox as eqx
import jax
from beartype.typing import Callable
from jaxtyping import Array, PRNGKeyArray

from jaxonmodels.layers.convolution import Conv2dNormActivation
from jaxonmodels.layers.normalization import BatchNorm
from jaxonmodels.layers.regularization import StochasticDepth


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

    def __call__(
        self,
        x: Array,
        activation: Callable[..., Array] = jax.nn.relu,
        scale_activation: Callable[..., Array] = jax.nn.sigmoid,
    ) -> Array:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = activation(scale)
        scale = self.fc2(scale)
        scale = scale_activation(scale)
        return scale * x


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
