import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxonlayers.layers import LayerNorm
from statedict2pytree import (
    convert,
    move_running_fields_to_the_end,
    pytree_to_fields,
    state_dict_to_fields,
)
from torchvision.models.swin_transformer import PatchMerging as TorchPatchMerging
from torchvision.models.swin_transformer import PatchMergingV2 as TorchPatchMergingV2
from torchvision.models.swin_transformer import (
    ShiftedWindowAttention as TorchShiftedWindowAttentionV1,
)
from torchvision.models.swin_transformer import (
    ShiftedWindowAttentionV2 as TorchShiftedWindowAttentionV2,
)
from torchvision.models.swin_transformer import SwinTransformer as TorchSwinTransformer
from torchvision.models.swin_transformer import (
    SwinTransformerBlock as TorchSwinTransformerBlock,
)
from torchvision.models.swin_transformer import (
    SwinTransformerBlockV2 as TorchSwinTransformerBlockV2,
)

from jaxonmodels.functions import default_floating_dtype
from jaxonmodels.models.swin_transformer import (
    PatchMerging,
    PatchMergingV2,
    ShiftedWindowAttention,
    ShiftedWindowAttentionV2,
    SwinTransformer,
    SwinTransformerBlock,
    SwinTransformerBlockV2,
)
from jaxonmodels.statedict2pytree import model_orders


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
        norm_layer=functools.partial(LayerNorm, dtype=dtype),
        inference=False,
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
        (96, 3, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.0, 56, 56),
        (192, 6, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0, 28, 28),
        (192, 6, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.0, 28, 28),
        (384, 12, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0, 14, 14),
        (384, 12, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.0, 14, 14),
        (384, 12, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0, 14, 14),
        (384, 12, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.0, 14, 14),
        (384, 12, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0, 14, 14),
        (384, 12, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.0, 14, 14),
        (768, 24, [7, 7], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0, 7, 7),
        (768, 24, [7, 7], [3, 3], 4.0, True, True, 0.0, 0.0, 0.0, 7, 7),
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
        norm_layer=functools.partial(eqx.nn.LayerNorm, eps=1e-05),  # pyright: ignore
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


@pytest.mark.parametrize(
    "dim, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, proj_bias, dropout, attention_dropout, stochastic_depth_prob, H, W",  # noqa
    [
        (96, 3, [8, 8], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0, 56, 56),
        (96, 3, [8, 8], [4, 4], 4.0, True, True, 0.0, 0.0, 0.0, 56, 56),
        (192, 6, [8, 8], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0, 28, 28),
        (192, 6, [8, 8], [4, 4], 4.0, True, True, 0.0, 0.0, 0.0, 28, 28),
        (384, 12, [8, 8], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0, 14, 14),
        (384, 12, [8, 8], [4, 4], 4.0, True, True, 0.0, 0.0, 0.0, 14, 14),
        (768, 24, [8, 8], [0, 0], 4.0, True, True, 0.0, 0.0, 0.0, 7, 7),
        (768, 24, [8, 8], [4, 4], 4.0, True, True, 0.0, 0.0, 0.0, 7, 7),
    ],
)
def test_swin_transformer_blockv2(
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
    torch_block = TorchSwinTransformerBlockV2(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attention_dropout=attention_dropout,
        stochastic_depth_prob=stochastic_depth_prob,
        norm_layer=torch.nn.LayerNorm,
        attn_layer=TorchShiftedWindowAttentionV2,
    )
    torch_block.eval()

    # --- JAX Model ---
    jax_block, state = eqx.nn.make_with_state(SwinTransformerBlockV2)(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attention_dropout=attention_dropout,
        stochastic_depth_prob=stochastic_depth_prob,
        norm_layer=functools.partial(eqx.nn.LayerNorm, eps=1e-05),  # pyright: ignore
        attn_layer=ShiftedWindowAttentionV2,
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
    torchfields = move_running_fields_to_the_end(
        torchfields,
        identifier="relative_coords_table",
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


def test_swinv1():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)

    patch_size = [4, 4]
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = [7, 7]
    mlp_ratio = 4.0
    dropout = 0.0
    attention_dropout = 0.0
    stochastic_depth_prob = 0.0
    num_classes = 1000

    torch_swin = TorchSwinTransformer(
        patch_size,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio,
        dropout,
        attention_dropout,
        stochastic_depth_prob,
        num_classes,
        block=TorchSwinTransformerBlock,
    )
    torch_swin.eval()

    x = np.array(np.random.normal(size=(1, 3, 224, 224)), dtype=np.float32)
    x_t = torch.Tensor(x)

    jax_swin, state = eqx.nn.make_with_state(SwinTransformer)(
        patch_size,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio,
        dropout,
        attention_dropout,
        stochastic_depth_prob,
        num_classes,
        norm_layer=None,
        block=SwinTransformerBlock,
        downsample_layer=PatchMerging,
        attn_layer=ShiftedWindowAttention,
        inference=True,
        key=jax.random.key(42),
    )

    weights_dict = torch_swin.state_dict()
    torchfields = state_dict_to_fields(weights_dict)

    jaxfields, state_indices = pytree_to_fields(
        (jax_swin, state),
        model_order=model_orders.get_swin_model_order(1),
    )
    jax_swin, state = convert(
        weights_dict,
        (jax_swin, state),
        jaxfields,
        state_indices,
        torchfields,
    )

    jax_swin, state = eqx.nn.inference_mode((jax_swin, state))
    key = jax.random.key(22)
    jax_swin_pt = functools.partial(jax_swin, key=key)
    jax_out, state = eqx.filter_vmap(
        jax_swin_pt, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(jnp.array(x), state)

    t_out = torch_swin.forward(x_t).detach().numpy()
    jax_out = np.array(jax_out)

    # Debug: print comparison info
    print(f"JAX out shape: {jax_out.shape}, first 5: {jax_out.flatten()[:5]}")
    print(f"Torch out shape: {t_out.shape}, first 5: {t_out.flatten()[:5]}")
    print(f"Input x first 5: {x.flatten()[:5]}")
    print(f"Max diff: {np.max(np.abs(jax_out - t_out))}")

    assert np.allclose(jax_out, t_out, atol=1e-3)


def test_swinv2():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)

    patch_size = [4, 4]
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = [7, 7]
    mlp_ratio = 4.0
    dropout = 0.0
    attention_dropout = 0.0
    stochastic_depth_prob = 0.0
    num_classes = 1000

    torch_swin = TorchSwinTransformer(
        patch_size,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio,
        dropout,
        attention_dropout,
        stochastic_depth_prob,
        num_classes,
        downsample_layer=TorchPatchMergingV2,
        block=TorchSwinTransformerBlockV2,
    )
    torch_swin.eval()

    x = np.array(np.random.normal(size=(1, 3, 224, 224)), dtype=np.float32)
    x_t = torch.Tensor(x)

    jax_swin, state = eqx.nn.make_with_state(SwinTransformer)(
        patch_size,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio,
        dropout,
        attention_dropout,
        stochastic_depth_prob,
        num_classes,
        norm_layer=None,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        attn_layer=ShiftedWindowAttentionV2,
        inference=True,
        key=jax.random.key(42),
    )

    weights_dict = torch_swin.state_dict()
    torchfields = state_dict_to_fields(weights_dict)
    jaxfields, state_indices = pytree_to_fields(
        (jax_swin, state),
        model_order=model_orders.get_swin_model_order(2),
    )

    jax_swin, state = convert(
        weights_dict,
        (jax_swin, state),
        jaxfields,
        state_indices,
        torchfields,
    )
    jax_swin, state = eqx.nn.inference_mode((jax_swin, state))

    key = jax.random.key(22)
    jax_swin_pt = functools.partial(jax_swin, key=key)
    jax_out, state = eqx.filter_vmap(
        jax_swin_pt, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(jnp.array(x), state)

    t_out = torch_swin.forward(x_t).detach().numpy()
    jax_out = np.array(jax_out)

    # Debug: print comparison info
    print(f"JAX out shape: {jax_out.shape}, first 5: {jax_out.flatten()[:5]}")
    print(f"Torch out shape: {t_out.shape}, first 5: {t_out.flatten()[:5]}")
    print(f"Input x first 5: {x.flatten()[:5]}")
    print(f"Max diff: {np.max(np.abs(jax_out - t_out))}")

    assert np.allclose(jax_out, t_out, atol=1e-3)
