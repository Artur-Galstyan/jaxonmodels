import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torchvision.models.swin_transformer import PatchMerging as TorchPatchMerging

from jaxonmodels.functions import default_floating_dtype
from jaxonmodels.layers import LayerNorm2d
from jaxonmodels.models.swin_transformer import PatchMerging
from jaxonmodels.statedict2pytree.s2p import (
    convert,
    move_running_fields_to_the_end,
    pytree_to_fields,
    state_dict_to_fields,
)


@pytest.mark.parametrize(
    "C, H, W",
    [
        (32, 55, 55),
        (32, 224, 224),
        (16, 56, 56),
        (32, 112, 56),
    ],
)
def test_patch_merging_layers(C, H, W):
    batch_size = 8
    dtype = default_floating_dtype()
    torch_patch = TorchPatchMerging(dim=C)
    jax_patch, state = eqx.nn.make_with_state(PatchMerging)(
        dim=C,
        norm_layer=functools.partial(LayerNorm2d, shape=4 * C, dtype=dtype),
        inference=False,
        axis_name=None,
        dtype=dtype,
        key=jax.random.key(42),
    )

    jax_patch, state = eqx.nn.inference_mode((jax_patch, state))

    weights_dict = torch_patch.state_dict()
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    jaxfields, state_indices = pytree_to_fields((jax_patch, state))

    jax_patch, state = convert(
        weights_dict,
        (jax_patch, state),
        jaxfields,
        state_indices,
        torchfields,
    )
    np.random.seed(42)
    x = np.array(np.random.normal(size=(batch_size, H, W, C)), dtype=np.float32)

    t_out = torch_patch(torch.from_numpy(x))
    j_out, state = eqx.filter_vmap(jax_patch, in_axes=(0, None), out_axes=(0, None))(
        jnp.array(x), state
    )

    assert np.allclose(np.array(j_out), t_out.detach().numpy(), atol=1e-6)
