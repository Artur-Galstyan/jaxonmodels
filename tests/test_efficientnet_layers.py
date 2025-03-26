import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchvision.ops.misc import Conv2dNormActivation as TorchConv2dNormActivation
from torchvision.ops.misc import SqueezeExcitation as TorchSqueezeExcitation

from jaxonmodels.layers.batch_norm import BatchNorm
from jaxonmodels.models.efficientnet import (  # , SqueezeExcitation
    Conv2dNormActivation,
    SqueezeExcitation,
)
from jaxonmodels.statedict2pytree.s2p import (
    convert,
    move_running_fields_to_the_end,
    pytree_to_fields,
    state_dict_to_fields,
)


def test_Conv2dNormActivation():
    key = jax.random.key(42)
    jax_conv2d_norm_act, state = eqx.nn.make_with_state(Conv2dNormActivation)(
        3,
        32,
        kernel_size=3,
        stride=2,
        norm_layer=BatchNorm,
        activation_layer=jax.nn.silu,
        key=key,
    )

    torch_conv2d_norm_act = TorchConv2dNormActivation(
        3,
        32,
        kernel_size=3,
        stride=2,
        norm_layer=torch.nn.BatchNorm2d,
        activation_layer=torch.nn.SiLU,
    )

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

    np.random.seed(42)
    x = np.array(np.random.normal(size=(4, 3, 224, 224)), dtype=np.float32)
    torch_conv2d_norm_act.eval()
    out_torch = torch_conv2d_norm_act.forward(torch.from_numpy(x))

    key, subkey = jax.random.split(key)
    jax_conv2d_norm_act_pt = functools.partial(
        jax_conv2d_norm_act, inference=True, key=subkey
    )

    out_jax, state = eqx.filter_vmap(
        jax_conv2d_norm_act_pt, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(jnp.array(x), state)

    assert np.allclose(out_torch.detach().numpy(), np.array(out_jax), atol=1e-6)


def test_SqueezeExcitation():
    input_channels = 96
    squeeze_channels = 4

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

    x = np.array(np.random.normal(size=(96, 112, 112)), dtype=np.float32)

    jax_output = jax_squeeze(jnp.array(x))
    torch_output = torch_squeeze(torch.from_numpy(x))

    assert np.allclose(np.array(jax_output), torch_output.detach().numpy(), atol=1e-6)
