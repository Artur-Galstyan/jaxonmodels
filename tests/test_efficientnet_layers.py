import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxonlayers.layers import BatchNorm, ConvNormActivation
from jaxonlayers.layers.attention import SqueezeExcitation
from statedict2pytree import (
    convert,
    move_running_fields_to_the_end,
    pytree_to_fields,
    state_dict_to_fields,
)
from torch.nn.modules import BatchNorm2d
from torchvision.models.efficientnet import MBConv as TorchMBConv
from torchvision.models.efficientnet import MBConvConfig as TorchMBConvConfig
from torchvision.ops.misc import Conv2dNormActivation as TorchConv2dNormActivation
from torchvision.ops.misc import SqueezeExcitation as TorchSqueezeExcitation

from jaxonmodels.functions.utils import default_floating_dtype
from jaxonmodels.models.efficientnet import (
    MBConv,
    MBConvConfig,
)


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, norm_layer, activation_layer",  # noqa
    [
        (3, 32, 3, 2, None, BatchNorm, jax.nn.silu),  # Base case
        (16, 24, 3, 1, None, BatchNorm, jax.nn.silu),  # Different channels
        (24, 40, 5, 2, 2, BatchNorm, jax.nn.silu),  # Larger kernel
        (40, 80, 3, 2, 1, BatchNorm, jax.nn.relu),  # Different activation
        (80, 112, 1, 1, 0, BatchNorm, jax.nn.silu),  # 1x1 conv
    ],
)
def test_Conv2dNormActivation_parametrized(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    norm_layer,
    activation_layer,
):
    dtype = default_floating_dtype()
    key = jax.random.key(42)

    # Create JAX version
    jax_conv2d_norm_act, state = eqx.nn.make_with_state(ConvNormActivation)(
        2,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        norm_layer=functools.partial(
            norm_layer, axis_name="batch", size=out_channels, dtype=dtype
        ),
        activation_layer=activation_layer,
        key=key,
        dtype=dtype,
    )

    # Create PyTorch version - convert JAX functions to torch equivalents
    torch_activation = (
        torch.nn.SiLU if activation_layer == jax.nn.silu else torch.nn.ReLU
    )
    torch_norm = torch.nn.BatchNorm2d if norm_layer == BatchNorm else None

    torch_conv2d_norm_act = TorchConv2dNormActivation(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        norm_layer=torch_norm,
        activation_layer=torch_activation,
    )

    # Convert weights
    weights_dict = torch_conv2d_norm_act.state_dict()
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    jaxfields, state_indices = pytree_to_fields((jax_conv2d_norm_act, state))

    jax_conv2d_norm_act, state = convert(
        weights_dict,
        (jax_conv2d_norm_act, state),
        jaxfields,
        state_indices,
        torchfields,
    )

    jax_conv2d_norm_act, state = eqx.nn.inference_mode((jax_conv2d_norm_act, state))

    # Generate random input
    np.random.seed(42)
    input_height, input_width = 64, 64  # Example dimensions
    x = np.array(
        np.random.normal(size=(4, in_channels, input_height, input_width)),
        dtype=np.float32,
    )

    # Prepare models for inference
    torch_conv2d_norm_act.eval()
    key, subkey = jax.random.split(key)
    jax_conv2d_norm_act_pt = functools.partial(jax_conv2d_norm_act, key=subkey)

    # Run inference
    out_torch = torch_conv2d_norm_act.forward(torch.from_numpy(x))
    out_jax, state = eqx.filter_vmap(
        jax_conv2d_norm_act_pt, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(jnp.array(x), state)

    # Verify outputs match
    assert np.allclose(out_torch.detach().numpy(), np.array(out_jax), atol=1e-5)


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, groups",
    [
        (32, 32, 3, 1, 32),  # Depthwise convolution
        (64, 64, 5, 2, 64),  # Depthwise with larger kernel and stride
    ],
)
def test_Conv2dNormActivation_groups(
    in_channels, out_channels, kernel_size, stride, groups
):
    dtype = default_floating_dtype()
    key = jax.random.key(42)

    # Create JAX version
    jax_conv2d_norm_act, state = eqx.nn.make_with_state(ConvNormActivation)(
        2,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        groups=groups,
        norm_layer=functools.partial(
            BatchNorm, axis_name="batch", size=out_channels, dtype=dtype
        ),
        activation_layer=jax.nn.silu,
        key=key,
        dtype=dtype,
    )

    # Create PyTorch version
    torch_conv2d_norm_act = TorchConv2dNormActivation(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        groups=groups,
        norm_layer=torch.nn.BatchNorm2d,
        activation_layer=torch.nn.SiLU,
    )

    # Convert weights
    weights_dict = torch_conv2d_norm_act.state_dict()
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    jaxfields, state_indices = pytree_to_fields((jax_conv2d_norm_act, state))

    jax_conv2d_norm_act, state = convert(
        weights_dict,
        (jax_conv2d_norm_act, state),
        jaxfields,
        state_indices,
        torchfields,
    )

    jax_conv2d_norm_act, state = eqx.nn.inference_mode((jax_conv2d_norm_act, state))

    # Generate random input
    np.random.seed(42)
    input_height, input_width = 32, 32  # Example dimensions
    x = np.array(
        np.random.normal(size=(2, in_channels, input_height, input_width)),
        dtype=np.float32,
    )

    # Prepare models for inference
    torch_conv2d_norm_act.eval()
    key, subkey = jax.random.split(key)
    jax_conv2d_norm_act_pt = functools.partial(jax_conv2d_norm_act, key=subkey)

    # Run inference
    out_torch = torch_conv2d_norm_act.forward(torch.from_numpy(x))
    out_jax, state = eqx.filter_vmap(
        jax_conv2d_norm_act_pt, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(jnp.array(x), state)

    # Verify outputs match
    assert np.allclose(out_torch.detach().numpy(), np.array(out_jax), atol=1e-6)


@pytest.mark.parametrize(
    "input_channels, squeeze_factor",
    [
        (96, 4),  # Original test case
        (16, 4),  # Smaller input channels
        (192, 8),  # Larger input channels, different squeeze factor
        (24, 2),  # Different combination
        (80, 16),  # Different combination
    ],
)
def test_SqueezeExcitation_parametrized(input_channels, squeeze_factor):
    squeeze_channels = max(1, input_channels // squeeze_factor)

    key = jax.random.key(42)
    jax_squeeze = SqueezeExcitation(input_channels, squeeze_channels, key=key)
    torch_squeeze = TorchSqueezeExcitation(input_channels, squeeze_channels)

    weights_dict = torch_squeeze.state_dict()
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    jaxfields, state_indices = pytree_to_fields(jax_squeeze)

    jax_squeeze = convert(
        weights_dict,
        jax_squeeze,
        jaxfields,
        state_indices,
        torchfields,
    )

    # Generate input with appropriate shape for testing
    np.random.seed(42)
    height, width = 28, 28  # Example dimensions
    x = np.array(
        np.random.normal(size=(input_channels, height, width)), dtype=np.float32
    )

    jax_output = jax_squeeze(jnp.array(x))
    torch_output = torch_squeeze(torch.from_numpy(x))

    assert np.allclose(np.array(jax_output), torch_output.detach().numpy(), atol=1e-6)


@pytest.mark.parametrize(
    "expand_ratio, kernel, stride, input_channels, out_channels, num_layers, stochastic_depth_prob, height, width",  # noqa
    [
        (1.0, 3, 1, 32, 16, 1, 0.0, 112, 112),
        (6.0, 3, 2, 16, 24, 2, 0.0125, 112, 112),
        (6.0, 3, 1, 24, 24, 2, 0.025, 56, 56),
        (6.0, 5, 2, 24, 40, 2, 0.0375, 56, 56),
        (6.0, 5, 1, 40, 40, 2, 0.05, 28, 28),
        (6.0, 3, 2, 40, 80, 3, 0.0625, 28, 28),
        (6.0, 3, 1, 80, 80, 3, 0.075, 14, 14),
        (6.0, 3, 1, 80, 80, 3, 0.0875, 14, 14),
        (6.0, 5, 1, 80, 112, 3, 0.1, 14, 14),
        (6.0, 5, 1, 112, 112, 3, 0.1125, 14, 14),
        (6.0, 5, 1, 112, 112, 3, 0.125, 14, 14),
        (6.0, 5, 2, 112, 192, 4, 0.1375, 14, 14),
        (6.0, 5, 1, 192, 192, 4, 0.15, 7, 7),
        (6.0, 5, 1, 192, 192, 4, 0.1625, 7, 7),
        (6.0, 5, 1, 192, 192, 4, 0.175, 7, 7),
        (6.0, 3, 1, 192, 320, 1, 0.1875, 7, 7),
    ],
)
def test_mbconv_forward(
    expand_ratio,
    kernel,
    stride,
    input_channels,
    out_channels,
    num_layers,
    stochastic_depth_prob,
    height,
    width,
):
    dtype = default_floating_dtype()
    t_cnf = TorchMBConvConfig(
        expand_ratio=expand_ratio,
        kernel=kernel,
        stride=stride,
        input_channels=input_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        block=TorchMBConv,
    )

    torch_mbconv = TorchMBConv(t_cnf, stochastic_depth_prob, BatchNorm2d)
    j_cnf = MBConvConfig(
        expand_ratio=expand_ratio,
        kernel=kernel,
        stride=stride,
        input_channels=input_channels,
        out_channels=out_channels,
        num_layers=num_layers,
    )
    key = jax.random.key(42)
    jax_mbconv, state = eqx.nn.make_with_state(MBConv)(
        j_cnf, stochastic_depth_prob, key=key, dtype=dtype
    )

    weights_dict = torch_mbconv.state_dict()
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    jaxfields, state_indices = pytree_to_fields((jax_mbconv, state))

    jax_mbconv, state = convert(
        weights_dict,
        (jax_mbconv, state),
        jaxfields,
        state_indices,
        torchfields,
    )

    jax_mbconv, state = eqx.nn.inference_mode((jax_mbconv, state))

    np.random.seed(42)
    x = np.array(
        np.random.normal(size=(2, input_channels, height, width)), dtype=np.float32
    )
    key, subkey = jax.random.split(key)
    jax_mbconv_pt = functools.partial(jax_mbconv, key=subkey)
    torch_mbconv.eval()

    jax_output, state = eqx.filter_vmap(
        jax_mbconv_pt, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(jnp.array(x), state)

    torch_output = torch_mbconv(torch.from_numpy(x))

    # print(jax_output[0])
    # print(torch_output[0])

    assert np.allclose(np.array(jax_output), torch_output.detach().numpy(), atol=1e-5)
