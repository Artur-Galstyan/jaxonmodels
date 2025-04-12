import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torchvision.models.swin_transformer import PatchMerging as TorchPatchMerging
from torchvision.models.swin_transformer import (
    ShiftedWindowAttention as TorchShiftedWindowAttentionV1,
)
from torchvision.models.swin_transformer import (
    ShiftedWindowAttentionV2 as TorchShiftedWindowAttentionV2,
)
from torchvision.models.swin_transformer import (
    SwinTransformerBlock as TorchSwinTransformerBlock,
)

from jaxonmodels.functions import default_floating_dtype
from jaxonmodels.layers import LayerNorm2d
from jaxonmodels.models.swin_transformer import (
    PatchMerging,
    ShiftedWindowAttention,
    ShiftedWindowAttentionV2,
    SwinTransformerBlock,
)
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


@pytest.mark.parametrize(
    "C, window_size, shift_size, num_heads, qkv_bias, proj_bias, attention_dropout, dropout, H, W",  # noqa
    [
        (96, [7, 7], [0, 0], 3, True, True, 0.0, 0.0, 56, 56),
        (96, [7, 7], [3, 3], 3, True, True, 0.0, 0.0, 56, 56),
        (192, [7, 7], [0, 0], 6, True, True, 0.0, 0.0, 56, 56),
        (192, [7, 7], [3, 3], 6, True, True, 0.0, 0.0, 56, 56),
        (384, [7, 7], [0, 0], 12, True, True, 0.0, 0.0, 56, 56),
        (384, [7, 7], [3, 3], 12, True, True, 0.0, 0.0, 56, 56),
        (768, [7, 7], [0, 0], 24, True, True, 0.0, 0.0, 56, 56),
        (768, [7, 7], [3, 3], 24, True, True, 0.0, 0.0, 56, 56),
        (96, [8, 8], [0, 0], 3, True, True, 0.0, 0.0, 56, 56),
        (96, [8, 8], [4, 4], 3, True, True, 0.0, 0.0, 56, 56),
        (192, [8, 8], [0, 0], 6, True, True, 0.0, 0.0, 56, 56),
        (192, [8, 8], [4, 4], 6, True, True, 0.0, 0.0, 56, 56),
        (384, [8, 8], [0, 0], 12, True, True, 0.0, 0.0, 56, 56),
        (384, [8, 8], [4, 4], 12, True, True, 0.0, 0.0, 56, 56),
        (768, [8, 8], [0, 0], 24, True, True, 0.0, 0.0, 56, 56),
        (768, [8, 8], [4, 4], 24, True, True, 0.0, 0.0, 56, 56),
    ],
)
def test_swin_attention_v1(
    C,
    window_size,
    shift_size,
    num_heads,
    qkv_bias,
    proj_bias,
    attention_dropout,
    dropout,
    H,
    W,
):
    batch_size = 8
    dtype = default_floating_dtype()
    torch_swin = TorchShiftedWindowAttentionV1(C, window_size, shift_size, num_heads)

    jax_swin, state = eqx.nn.make_with_state(ShiftedWindowAttention)(
        dim=C,
        shift_size=shift_size,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        window_size=window_size,
        proj_bias=proj_bias,
        attention_dropout=attention_dropout,
        dropout=dropout,
        inference=False,
        dtype=dtype,
        key=jax.random.key(42),
    )

    jax_swin, state = eqx.nn.inference_mode((jax_swin, state))

    weights_dict = torch_swin.state_dict()
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    torchfields = move_running_fields_to_the_end(
        torchfields, identifier="relative_position_index"
    )
    jaxfields, state_indices = pytree_to_fields((jax_swin, state))

    # for t, j in zip(torchfields, jaxfields):
    #     print(t.path, t.shape, jax.tree_util.keystr(j.path), j.shape)

    jax_swin, state = convert(
        weights_dict,
        (jax_swin, state),
        jaxfields,
        state_indices,
        torchfields,
    )

    np.random.seed(42)
    x = np.array(np.random.normal(size=(batch_size, H, W, C)), dtype=np.float32)

    t_out = torch_swin(torch.from_numpy(x))

    j_out, state = eqx.filter_vmap(jax_swin, in_axes=(0, None), out_axes=(0, None))(
        jnp.array(x), state
    )

    assert np.allclose(np.array(j_out), t_out.detach().numpy(), atol=1e-5)


@pytest.mark.parametrize(
    "C, window_size, shift_size, num_heads, qkv_bias, proj_bias, attention_dropout, dropout, H, W",  # noqa
    [
        (96, [8, 8], [0, 0], 3, True, True, 0.0, 0.0, 56, 56),
        (96, [8, 8], [4, 4], 3, True, True, 0.0, 0.0, 56, 56),
        (192, [8, 8], [0, 0], 6, True, True, 0.0, 0.0, 56, 56),
        (192, [8, 8], [4, 4], 6, True, True, 0.0, 0.0, 56, 56),
        (384, [8, 8], [0, 0], 12, True, True, 0.0, 0.0, 56, 56),
        (384, [8, 8], [4, 4], 12, True, True, 0.0, 0.0, 56, 56),
        (384, [8, 8], [0, 0], 12, True, True, 0.0, 0.0, 56, 56),
        (384, [8, 8], [4, 4], 12, True, True, 0.0, 0.0, 56, 56),
        (384, [8, 8], [0, 0], 12, True, True, 0.0, 0.0, 56, 56),
        (384, [8, 8], [4, 4], 12, True, True, 0.0, 0.0, 56, 56),
        (768, [8, 8], [0, 0], 24, True, True, 0.0, 0.0, 56, 56),
        (768, [8, 8], [4, 4], 24, True, True, 0.0, 0.0, 56, 56),
    ],
)
def test_swin_attention_v2(
    C,
    window_size,
    shift_size,
    num_heads,
    qkv_bias,
    proj_bias,
    attention_dropout,
    dropout,
    H,
    W,
):
    batch_size = 8
    dtype = default_floating_dtype()
    torch_swin = TorchShiftedWindowAttentionV2(C, window_size, shift_size, num_heads)

    jax_swin, state = eqx.nn.make_with_state(ShiftedWindowAttentionV2)(
        dim=C,
        shift_size=shift_size,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        window_size=window_size,
        proj_bias=proj_bias,
        attention_dropout=attention_dropout,
        dropout=dropout,
        inference=False,
        dtype=dtype,
        key=jax.random.key(42),
    )

    jax_swin, state = eqx.nn.inference_mode((jax_swin, state))

    weights_dict = torch_swin.state_dict()
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    torchfields = move_running_fields_to_the_end(
        torchfields, identifier="relative_position_index"
    )
    torchfields = move_running_fields_to_the_end(
        torchfields, identifier="relative_coords_table"
    )

    jaxfields, state_indices = pytree_to_fields((jax_swin, state))

    for t, j in zip(torchfields, jaxfields):
        print(t.path, t.shape, jax.tree_util.keystr(j.path), j.shape)

    jax_swin, state = convert(
        weights_dict,
        (jax_swin, state),
        jaxfields,
        state_indices,
        torchfields,
    )

    np.random.seed(42)
    x = np.array(np.random.normal(size=(batch_size, H, W, C)), dtype=np.float32)

    t_out = torch_swin(torch.from_numpy(x))

    j_out, state = eqx.filter_vmap(jax_swin, in_axes=(0, None), out_axes=(0, None))(
        jnp.array(x), state
    )

    assert np.allclose(np.array(j_out), t_out.detach().numpy(), atol=1e-5)


@pytest.mark.parametrize(
    "dim, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, proj_bias, dropout, attention_dropout, stochastic_depth_prob, H, W",  # noqa
    [
        (96, 3, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0, 56, 56),
        (96, 3, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.01818, 56, 56),
        (192, 6, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0363636, 28, 28),
        (192, 6, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.0545456, 28, 28),
        (384, 12, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.072727, 14, 14),
        (384, 12, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.090909, 14, 14),
        (384, 12, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.109090, 14, 14),
        (384, 12, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.127272, 14, 14),
        (384, 12, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.145454, 14, 14),
        (384, 12, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.163636, 14, 14),
        (768, 24, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.181818, 7, 7),
        (768, 24, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.2, 7, 7),
    ],
)
def test_swin_transformer_block(
    dim,
    num_heads,
    window_size,
    shift_size,
    mlp_ratio,
    qkv_bias,
    proj_bias,
    dropout,
    attention_dropout,
    stochastic_depth_prob,
    H,
    W,
):
    batch_size = 2
    dtype = default_floating_dtype()
    key = jax.random.key(43)
    torch.manual_seed(42)
    np.random.seed(42)

    # --- PyTorch Model ---

    torch_block = TorchSwinTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attention_dropout=attention_dropout,
        stochastic_depth_prob=stochastic_depth_prob,
        norm_layer=torch.nn.LayerNorm,
        attn_layer=TorchShiftedWindowAttentionV1,
    )
    torch_block.eval()

    # --- JAX Model ---
    #  Always use the V1 attention module.
    jax_block, state = eqx.nn.make_with_state(SwinTransformerBlock)(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attention_dropout=attention_dropout,
        stochastic_depth_prob=stochastic_depth_prob,
        norm_layer=functools.partial(eqx.nn.LayerNorm, eps=1e-05),
        attn_layer=ShiftedWindowAttention,
        inference=False,
        dtype=dtype,
        key=key,
    )
    jax_block, state = eqx.nn.inference_mode((jax_block, state))

    # --- Weight Conversion ---
    weights_dict = torch_block.state_dict()
    torchfields = state_dict_to_fields(weights_dict)
    torchfields = move_running_fields_to_the_end(torchfields)
    torchfields = move_running_fields_to_the_end(
        torchfields,
        identifier="relative_position_index",
    )

    jaxfields, state_indices = pytree_to_fields(
        (
            jax_block,
            state,
        )
    )

    jax_block, state = convert(
        weights_dict,
        (jax_block, state),
        jaxfields,
        state_indices,
        torchfields,
    )

    # --- Inference and Comparison ---
    x_np = np.array(np.random.uniform(size=(batch_size, H, W, dim)), dtype=np.float32)
    x_torch = torch.from_numpy(x_np)
    x_jax = jnp.array(x_np)

    t_out = torch_block(x_torch)
    jax_block_pt = functools.partial(jax_block, key=jax.random.key(2100))
    j_out, state = eqx.filter_vmap(jax_block_pt, in_axes=(0, None), out_axes=(0, None))(
        x_jax, state
    )

    assert np.allclose(np.array(j_out), t_out.detach().numpy(), atol=1e-3)
