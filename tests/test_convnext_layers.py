import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from equinox.nn import LayerNorm
from torchvision.models.convnext import CNBlock as TorchCNBlock

from jaxonmodels.models.convnext import CNBlock
from jaxonmodels.statedict2pytree.s2p import (
    convert,
    move_running_fields_to_the_end,
    pytree_to_fields,
    state_dict_to_fields,
)


def test_cnblock_equivalence():
    batch_size = 4
    dim = 128
    layer_scale = 1e-06
    stochastic_depth_prob = 0.0
    norm_layer = None

    key = jax.random.PRNGKey(42)
    eqx_init_key, data_key, vmap_key = jax.random.split(key, 3)

    dtype = jnp.float32
    np_dtype = np.float32
    torch_dtype = torch.float32

    torch_block = TorchCNBlock(
        dim, layer_scale, stochastic_depth_prob, norm_layer=norm_layer
    )

    jax_block, state = eqx.nn.make_with_state(CNBlock)(
        dim,
        layer_scale,
        stochastic_depth_prob,
        norm_layer=LayerNorm,
        key=eqx_init_key,
        dtype=dtype,
    )

    weights_dict = torch_block.state_dict()
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    jaxfields, state_indices = pytree_to_fields((jax_block, state))

    # Use the convert function
    jax_block, state = convert(
        weights_dict,
        (jax_block, state),
        jaxfields,
        state_indices,
        torchfields,
        dtype=dtype,
    )

    jax_block, state = eqx.nn.inference_mode((jax_block, state))

    np_input = np.random.rand(batch_size, dim, 56, 56).astype(np_dtype)
    torch_input = torch.from_numpy(np_input).to(torch_dtype)

    torch_block.eval()
    with torch.no_grad():
        torch_output = torch_block(torch_input)

    jax_input = jnp.array(np_input, dtype=dtype)
    jax_infer_func = functools.partial(jax_block, key=vmap_key)
    jax_output = eqx.filter_vmap(jax_infer_func)(jax_input)

    assert jax_output.shape == torch_output.shape
    assert np.allclose(np.array(jax_output), torch_output.numpy(), atol=1e-5)
