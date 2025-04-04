import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest  # <-- Import pytest
import torch
from torchvision.models.convnext import CNBlock as TorchCNBlock

from jaxonmodels.models.convnext import CNBlock
from jaxonmodels.statedict2pytree.s2p import (
    convert,
    move_running_fields_to_the_end,
    pytree_to_fields,
    state_dict_to_fields,
)


# Define the different parameter sets you want to test
@pytest.mark.parametrize(
    "batch_size, dim, layer_scale, stochastic_depth_prob, height, width",
    [
        # Original parameters
        (4, 128, 1e-6, 0.0, 56, 56),
        # Different dimension
        (4, 64, 1e-6, 0.0, 56, 56),
        # Different layer scale
        (4, 128, 1e-4, 0.0, 56, 56),
        # With stochastic depth
        (2, 128, 1e-6, 0.1, 56, 56),
        # Different input size
        (4, 128, 1e-6, 0.0, 28, 28),
        # Single batch item
        (1, 96, 1e-6, 0.0, 40, 40),
        # Combined changes
        (2, 64, 1e-5, 0.05, 32, 32),
    ],
)
def test_cnblock_equivalence_parametrized(  # Renamed function slightly
    batch_size,
    dim,
    layer_scale,
    stochastic_depth_prob,
    height,
    width,  # Added params as args
):
    # Keep norm_layer fixed as requested
    norm_layer_torch = None
    norm_layer_eqx = None

    key = jax.random.PRNGKey(42)
    # Use different keys for each test run potentially by splitting from a base key
    # Or reuse like this if deterministic randomness per parameter set is okay.
    eqx_init_key, data_key, vmap_key = jax.random.split(key, 3)

    dtype = jnp.float32
    np_dtype = np.float32
    torch_dtype = torch.float32

    # Use the parameters in the model instantiation
    torch_block = TorchCNBlock(
        dim, layer_scale, stochastic_depth_prob, norm_layer=norm_layer_torch
    )

    jax_block, state = eqx.nn.make_with_state(CNBlock)(
        dim,
        layer_scale,
        stochastic_depth_prob,
        norm_layer=norm_layer_eqx,  # Keep Equinox LayerNorm
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

    # Use parameters for input shape
    np_input = np.random.rand(batch_size, dim, height, width).astype(np_dtype)
    torch_input = torch.from_numpy(np_input).to(torch_dtype)

    torch_block.eval()
    with torch.no_grad():
        torch_output = torch_block(torch_input)

    jax_input = jnp.array(np_input, dtype=dtype)
    # Ensure key is handled correctly if needed per-batch-item
    # If the model uses dropout/rng during inference based on the key,
    # you might need a different key handling strategy for vmap
    # For standard inference CNBlock, this partial should be fine.
    jax_infer_func = functools.partial(
        jax_block, key=None
    )  # Usually key is for dropout during training
    jax_output = eqx.filter_vmap(jax_infer_func)(jax_input)

    assert jax_output.shape == torch_output.shape
    # You might need to adjust tolerance (atol/rtol) depending on params
    assert np.allclose(np.array(jax_output), torch_output.numpy(), atol=1e-5, rtol=1e-5)
