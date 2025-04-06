import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torchvision.models.swin_transformer import (
    shifted_window_attention as torch_shifted_window_attention,
)

from jaxonmodels.functions import multi_head_attention_forward, shifted_window_attention


def test_compare_mha_with_pytorch():
    """Compare JAX multi-head attention implementation with PyTorch's."""

    batch_size = 2  # Only for PyTorch
    d_model = 16
    num_heads = 4
    tgt_len = 6
    src_len = 8

    np.random.seed(42)

    pt_query = np.random.rand(tgt_len, batch_size, d_model)
    pt_key = np.random.rand(src_len, batch_size, d_model)
    pt_value = np.random.rand(src_len, batch_size, d_model)

    jax_query = jnp.transpose(jnp.array(pt_query), (1, 0, 2))
    jax_key = jnp.transpose(jnp.array(pt_key), (1, 0, 2))
    jax_value = jnp.transpose(jnp.array(pt_value), (1, 0, 2))

    pt_in_proj_weight = torch.rand(3 * d_model, d_model)
    pt_in_proj_bias = torch.rand(3 * d_model)
    pt_out_proj_weight = torch.rand(d_model, d_model)
    pt_out_proj_bias = torch.rand(d_model)

    jax_in_proj_weight = jnp.array(pt_in_proj_weight.numpy())
    jax_in_proj_bias = jnp.array(pt_in_proj_bias.numpy())
    jax_out_proj_weight = jnp.array(pt_out_proj_weight.numpy())
    jax_out_proj_bias = jnp.array(pt_out_proj_bias.numpy())

    pt_attn_mask = torch.zeros(tgt_len, src_len)
    jax_attn_mask = jnp.array(pt_attn_mask.numpy())

    pt_output, pt_attn_weights = F.multi_head_attention_forward(
        query=torch.Tensor(pt_query),
        key=torch.Tensor(pt_key),
        value=torch.Tensor(pt_value),
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=pt_in_proj_weight,
        in_proj_bias=pt_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=pt_out_proj_weight,
        out_proj_bias=pt_out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=pt_attn_mask,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    pt_output_single = pt_output.detach().numpy()

    partial_mha = ft.partial(
        multi_head_attention_forward,
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=jax_in_proj_weight,
        in_proj_bias=jax_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=jax_out_proj_weight,
        out_proj_bias=jax_out_proj_bias,
        inference=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=jax_attn_mask,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
        dropout_key=None,
    )

    jax_output, jax_attn_weights = eqx.filter_vmap(partial_mha, out_axes=(1, 0))(
        jax_query,
        jax_key,
        jax_value,
    )

    output_close = np.allclose(pt_output_single, jax_output, rtol=1e-5, atol=1e-5)
    assert output_close

    assert pt_attn_weights is not None

    attn_close = np.allclose(pt_attn_weights, jax_attn_weights, rtol=1e-5, atol=1e-5)
    assert attn_close


def test_compare_mha_with_different_dimensions():
    """Test with different dimensions for model size and sequence lengths."""
    batch_size = 3  # Only for PyTorch
    d_model = 32
    num_heads = 8
    tgt_len = 10
    src_len = 12

    np.random.seed(43)
    pt_query = np.random.rand(tgt_len, batch_size, d_model)
    pt_key = np.random.rand(src_len, batch_size, d_model)
    pt_value = np.random.rand(src_len, batch_size, d_model)

    jax_query = jnp.transpose(jnp.array(pt_query), (1, 0, 2))
    jax_key = jnp.transpose(jnp.array(pt_key), (1, 0, 2))
    jax_value = jnp.transpose(jnp.array(pt_value), (1, 0, 2))

    pt_in_proj_weight = torch.rand(3 * d_model, d_model)
    pt_in_proj_bias = torch.rand(3 * d_model)
    pt_out_proj_weight = torch.rand(d_model, d_model)
    pt_out_proj_bias = torch.rand(d_model)

    jax_in_proj_weight = jnp.array(pt_in_proj_weight.numpy())
    jax_in_proj_bias = jnp.array(pt_in_proj_bias.numpy())
    jax_out_proj_weight = jnp.array(pt_out_proj_weight.numpy())
    jax_out_proj_bias = jnp.array(pt_out_proj_bias.numpy())

    pt_output, pt_attn_weights = F.multi_head_attention_forward(
        query=torch.Tensor(pt_query),
        key=torch.Tensor(pt_key),
        value=torch.Tensor(pt_value),
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=pt_in_proj_weight,
        in_proj_bias=pt_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=pt_out_proj_weight,
        out_proj_bias=pt_out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    pt_output_single = pt_output.detach().numpy()

    partial_mha = ft.partial(
        multi_head_attention_forward,
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=jax_in_proj_weight,
        in_proj_bias=jax_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=jax_out_proj_weight,
        out_proj_bias=jax_out_proj_bias,
        inference=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
        dropout_key=None,
    )

    jax_output, jax_attn_weights = eqx.filter_vmap(partial_mha, out_axes=(1, 0))(
        jax_query,
        jax_key,
        jax_value,
    )

    output_close = np.allclose(pt_output_single, jax_output, rtol=1e-5, atol=1e-5)
    assert output_close

    assert pt_attn_weights is not None
    attn_close = np.allclose(pt_attn_weights, jax_attn_weights, rtol=1e-5, atol=1e-5)
    assert attn_close


def test_mha_with_causal_mask():
    """Test multi-head attention with causal masking."""
    batch_size = 2  # Only for PyTorch
    d_model = 16
    num_heads = 4
    seq_len = 10  # Same length for query and key/value for causal masking

    np.random.seed(44)
    pt_query = np.random.rand(seq_len, batch_size, d_model)
    pt_key = np.random.rand(seq_len, batch_size, d_model)
    pt_value = np.random.rand(seq_len, batch_size, d_model)

    jax_query = jnp.transpose(jnp.array(pt_query), (1, 0, 2))
    jax_key = jnp.transpose(jnp.array(pt_key), (1, 0, 2))
    jax_value = jnp.transpose(jnp.array(pt_value), (1, 0, 2))

    pt_in_proj_weight = torch.rand(3 * d_model, d_model)
    pt_in_proj_bias = torch.rand(3 * d_model)
    pt_out_proj_weight = torch.rand(d_model, d_model)
    pt_out_proj_bias = torch.rand(d_model)

    jax_in_proj_weight = jnp.array(pt_in_proj_weight.numpy())
    jax_in_proj_bias = jnp.array(pt_in_proj_bias.numpy())
    jax_out_proj_weight = jnp.array(pt_out_proj_weight.numpy())
    jax_out_proj_bias = jnp.array(pt_out_proj_bias.numpy())

    # Create causal mask for PyTorch (upper triangular with -inf)
    pt_causal_mask = torch.triu(
        torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1
    )
    jax_causal_mask = jnp.array(pt_causal_mask.numpy())

    # With causal masking
    pt_output, pt_attn_weights = F.multi_head_attention_forward(
        query=torch.Tensor(pt_query),
        key=torch.Tensor(pt_key),
        value=torch.Tensor(pt_value),
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=pt_in_proj_weight,
        in_proj_bias=pt_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=pt_out_proj_weight,
        out_proj_bias=pt_out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=pt_causal_mask,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,  # Set to False since we're using explicit mask
    )

    pt_output_single = pt_output.detach().numpy()

    partial_mha = ft.partial(
        multi_head_attention_forward,
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=jax_in_proj_weight,
        in_proj_bias=jax_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=jax_out_proj_weight,
        out_proj_bias=jax_out_proj_bias,
        inference=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=jax_causal_mask,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,  # Using explicit mask instead
        dropout_key=None,
    )

    jax_output, jax_attn_weights = eqx.filter_vmap(partial_mha, out_axes=(1, 0))(
        jax_query,
        jax_key,
        jax_value,
    )

    output_close = np.allclose(pt_output_single, jax_output, rtol=1e-5, atol=1e-5)
    assert output_close

    assert pt_attn_weights is not None

    # Check if the causal mask was correctly applied by ensuring upper triangle is zero
    attn_weights_np = np.array(jax_attn_weights)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert np.allclose(attn_weights_np[:, i, j], 0.0, atol=1e-7)

    attn_close = np.allclose(pt_attn_weights, jax_attn_weights, rtol=1e-5, atol=1e-5)
    assert attn_close


def test_mha_with_key_padding_mask():
    """Test multi-head attention with key padding mask."""
    batch_size = 2  # Only for PyTorch
    d_model = 16
    num_heads = 4
    tgt_len = 6
    src_len = 8

    np.random.seed(45)
    pt_query = np.random.rand(tgt_len, batch_size, d_model)
    pt_key = np.random.rand(src_len, batch_size, d_model)
    pt_value = np.random.rand(src_len, batch_size, d_model)

    jax_query = jnp.transpose(jnp.array(pt_query), (1, 0, 2))
    jax_key = jnp.transpose(jnp.array(pt_key), (1, 0, 2))
    jax_value = jnp.transpose(jnp.array(pt_value), (1, 0, 2))

    pt_in_proj_weight = torch.rand(3 * d_model, d_model)
    pt_in_proj_bias = torch.rand(3 * d_model)
    pt_out_proj_weight = torch.rand(d_model, d_model)
    pt_out_proj_bias = torch.rand(d_model)

    jax_in_proj_weight = jnp.array(pt_in_proj_weight.numpy())
    jax_in_proj_bias = jnp.array(pt_in_proj_bias.numpy())
    jax_out_proj_weight = jnp.array(pt_out_proj_weight.numpy())
    jax_out_proj_bias = jnp.array(pt_out_proj_bias.numpy())

    # Create padding masks (True means position is padded)
    # We'll mask the last 2 positions in each sequence
    pt_key_padding_mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
    pt_key_padding_mask[:, -2:] = True

    jax_key_padding_masks = []
    for i in range(batch_size):
        mask = jnp.zeros(src_len, dtype=bool)
        mask = mask.at[-2:].set(True)
        jax_key_padding_masks.append(mask)

    pt_output, pt_attn_weights = F.multi_head_attention_forward(
        query=torch.Tensor(pt_query),
        key=torch.Tensor(pt_key),
        value=torch.Tensor(pt_value),
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=pt_in_proj_weight,
        in_proj_bias=pt_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=pt_out_proj_weight,
        out_proj_bias=pt_out_proj_bias,
        training=False,
        key_padding_mask=pt_key_padding_mask,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    pt_output_single = pt_output.detach().numpy()

    # Need to run separately for each batch since our padding mask is per-batch
    jax_outputs = []
    jax_attn_weights_all = []

    for i in range(batch_size):
        partial_mha = ft.partial(
            multi_head_attention_forward,
            embed_dim_to_check=d_model,
            num_heads=num_heads,
            in_proj_weight=jax_in_proj_weight,
            in_proj_bias=jax_in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=jax_out_proj_weight,
            out_proj_bias=jax_out_proj_bias,
            inference=True,
            key_padding_mask=jax_key_padding_masks[i],
            need_weights=True,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
            dropout_key=None,
        )

        jax_output, jax_attn_weights = partial_mha(
            jax_query[i],
            jax_key[i],
            jax_value[i],
        )

        jax_outputs.append(jax_output)
        jax_attn_weights_all.append(jax_attn_weights)

    # Stack the results to match PyTorch's output shape
    jax_output_stacked = jnp.stack(jax_outputs, axis=1)

    output_close = np.allclose(
        pt_output_single, jax_output_stacked, rtol=1e-5, atol=1e-5
    )
    assert output_close

    # Verify that attention to padded positions is zeroed out
    for i in range(batch_size):
        attn_weights = jax_attn_weights_all[i]
        # Check that all attention weights to the last two positions are close to zero
        assert np.allclose(attn_weights[:, -2:], 0.0, atol=1e-5)


def test_mha_with_separate_projection_weights():
    """Test multi-head attention with separate projection weights for Q, K, V."""
    batch_size = 2  # Only for PyTorch
    d_model = 16
    num_heads = 4
    tgt_len = 6
    src_len = 8

    np.random.seed(46)
    pt_query = np.random.rand(tgt_len, batch_size, d_model)
    pt_key = np.random.rand(src_len, batch_size, d_model)
    pt_value = np.random.rand(src_len, batch_size, d_model)

    jax_query = jnp.transpose(jnp.array(pt_query), (1, 0, 2))
    jax_key = jnp.transpose(jnp.array(pt_key), (1, 0, 2))
    jax_value = jnp.transpose(jnp.array(pt_value), (1, 0, 2))

    # Separate projection weights for Q, K, V
    pt_q_proj_weight = torch.rand(d_model, d_model)
    pt_k_proj_weight = torch.rand(d_model, d_model)
    pt_v_proj_weight = torch.rand(d_model, d_model)
    pt_in_proj_bias = torch.rand(3 * d_model)
    pt_out_proj_weight = torch.rand(d_model, d_model)
    pt_out_proj_bias = torch.rand(d_model)

    jax_q_proj_weight = jnp.array(pt_q_proj_weight.numpy())
    jax_k_proj_weight = jnp.array(pt_k_proj_weight.numpy())
    jax_v_proj_weight = jnp.array(pt_v_proj_weight.numpy())
    jax_in_proj_bias = jnp.array(pt_in_proj_bias.numpy())
    jax_out_proj_weight = jnp.array(pt_out_proj_weight.numpy())
    jax_out_proj_bias = jnp.array(pt_out_proj_bias.numpy())

    pt_output, pt_attn_weights = F.multi_head_attention_forward(
        query=torch.Tensor(pt_query),
        key=torch.Tensor(pt_key),
        value=torch.Tensor(pt_value),
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=None,
        in_proj_bias=pt_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=pt_out_proj_weight,
        out_proj_bias=pt_out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=True,
        q_proj_weight=pt_q_proj_weight,
        k_proj_weight=pt_k_proj_weight,
        v_proj_weight=pt_v_proj_weight,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    pt_output_single = pt_output.detach().numpy()

    partial_mha = ft.partial(
        multi_head_attention_forward,
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=None,
        in_proj_bias=jax_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=jax_out_proj_weight,
        out_proj_bias=jax_out_proj_bias,
        inference=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=True,
        q_proj_weight=jax_q_proj_weight,
        k_proj_weight=jax_k_proj_weight,
        v_proj_weight=jax_v_proj_weight,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
        dropout_key=None,
    )

    jax_output, jax_attn_weights = eqx.filter_vmap(partial_mha, out_axes=(1, 0))(
        jax_query,
        jax_key,
        jax_value,
    )

    output_close = np.allclose(pt_output_single, jax_output, rtol=1e-5, atol=1e-5)
    assert output_close

    assert pt_attn_weights is not None
    attn_close = np.allclose(pt_attn_weights, jax_attn_weights, rtol=1e-5, atol=1e-5)
    assert attn_close


def test_mha_with_add_zero_attn():
    """Test multi-head attention with add_zero_attn option."""
    batch_size = 2  # Only for PyTorch
    d_model = 16
    num_heads = 4
    tgt_len = 6
    src_len = 8

    np.random.seed(47)
    pt_query = np.random.rand(tgt_len, batch_size, d_model)
    pt_key = np.random.rand(src_len, batch_size, d_model)
    pt_value = np.random.rand(src_len, batch_size, d_model)

    jax_query = jnp.transpose(jnp.array(pt_query), (1, 0, 2))
    jax_key = jnp.transpose(jnp.array(pt_key), (1, 0, 2))
    jax_value = jnp.transpose(jnp.array(pt_value), (1, 0, 2))

    pt_in_proj_weight = torch.rand(3 * d_model, d_model)
    pt_in_proj_bias = torch.rand(3 * d_model)
    pt_out_proj_weight = torch.rand(d_model, d_model)
    pt_out_proj_bias = torch.rand(d_model)

    jax_in_proj_weight = jnp.array(pt_in_proj_weight.numpy())
    jax_in_proj_bias = jnp.array(pt_in_proj_bias.numpy())
    jax_out_proj_weight = jnp.array(pt_out_proj_weight.numpy())
    jax_out_proj_bias = jnp.array(pt_out_proj_bias.numpy())

    # With add_zero_attn=True
    pt_output, pt_attn_weights = F.multi_head_attention_forward(
        query=torch.Tensor(pt_query),
        key=torch.Tensor(pt_key),
        value=torch.Tensor(pt_value),
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=pt_in_proj_weight,
        in_proj_bias=pt_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=True,  # Add zero attention
        dropout_p=0.0,
        out_proj_weight=pt_out_proj_weight,
        out_proj_bias=pt_out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    pt_output_single = pt_output.detach().numpy()

    partial_mha = ft.partial(
        multi_head_attention_forward,
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=jax_in_proj_weight,
        in_proj_bias=jax_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=True,  # Add zero attention
        dropout_p=0.0,
        out_proj_weight=jax_out_proj_weight,
        out_proj_bias=jax_out_proj_bias,
        inference=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
        dropout_key=None,
    )

    jax_output, jax_attn_weights = eqx.filter_vmap(partial_mha, out_axes=(1, 0))(
        jax_query,
        jax_key,
        jax_value,
    )

    output_close = np.allclose(pt_output_single, jax_output, rtol=1e-5, atol=1e-5)
    assert output_close

    assert pt_attn_weights is not None
    # PyTorch's attention weights will include the added zero attention dimension
    # Shape should be (batch_size, tgt_len, src_len + 1)
    assert pt_attn_weights.shape[-1] == src_len + 1
    assert jax_attn_weights.shape[-1] == src_len + 1

    attn_close = np.allclose(pt_attn_weights, jax_attn_weights, rtol=1e-5, atol=1e-5)
    assert attn_close


def test_mha_mixed_queries_and_kvs():
    """Test multi-head attention with different query and key/value dimensions."""
    # Using same batch size but different sequence lengths
    batch_size = 2
    d_model = 16
    num_heads = 4
    tgt_len = 6
    src_len = 8

    np.random.seed(50)
    # Create query, key, value with the same batch size but different sequence lengths
    pt_query = torch.tensor(
        np.random.rand(tgt_len, batch_size, d_model), dtype=torch.float32
    )
    pt_key = torch.tensor(
        np.random.rand(src_len, batch_size, d_model), dtype=torch.float32
    )
    pt_value = torch.tensor(
        np.random.rand(src_len, batch_size, d_model), dtype=torch.float32
    )

    # Convert to JAX format (batch first)
    jax_query = jnp.transpose(jnp.array(pt_query.numpy()), (1, 0, 2))
    jax_key = jnp.transpose(jnp.array(pt_key.numpy()), (1, 0, 2))
    jax_value = jnp.transpose(jnp.array(pt_value.numpy()), (1, 0, 2))

    # Create weights and biases
    pt_in_proj_weight = torch.tensor(
        np.random.rand(3 * d_model, d_model), dtype=torch.float32
    )
    pt_in_proj_bias = torch.tensor(np.random.rand(3 * d_model), dtype=torch.float32)
    pt_out_proj_weight = torch.tensor(
        np.random.rand(d_model, d_model), dtype=torch.float32
    )
    pt_out_proj_bias = torch.tensor(np.random.rand(d_model), dtype=torch.float32)

    jax_in_proj_weight = jnp.array(pt_in_proj_weight.numpy())
    jax_in_proj_bias = jnp.array(pt_in_proj_bias.numpy())
    jax_out_proj_weight = jnp.array(pt_out_proj_weight.numpy())
    jax_out_proj_bias = jnp.array(pt_out_proj_bias.numpy())

    # Run PyTorch implementation
    pt_output, pt_attn_weights = F.multi_head_attention_forward(
        query=pt_query,
        key=pt_key,
        value=pt_value,
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=pt_in_proj_weight,
        in_proj_bias=pt_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=pt_out_proj_weight,
        out_proj_bias=pt_out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    pt_output_numpy = pt_output.detach().numpy()

    # Setup JAX implementation
    partial_mha = ft.partial(
        multi_head_attention_forward,
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=jax_in_proj_weight,
        in_proj_bias=jax_in_proj_bias,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=jax_out_proj_weight,
        out_proj_bias=jax_out_proj_bias,
        inference=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
        dropout_key=None,
    )

    # Run JAX implementation with vmap
    jax_output, jax_attn_weights = eqx.filter_vmap(partial_mha, out_axes=(1, 0))(
        jax_query,
        jax_key,
        jax_value,
    )

    # Compare outputs
    output_close = np.allclose(pt_output_numpy, jax_output, rtol=1e-5, atol=1e-5)
    assert output_close

    # Compare attention weights
    assert pt_attn_weights is not None
    attn_close = np.allclose(
        pt_attn_weights.numpy(), jax_attn_weights, rtol=1e-5, atol=1e-5
    )
    assert attn_close


def test_mha_with_bias_kv():
    """Test multi-head attention with bias_k and bias_v."""
    batch_size = 2
    d_model = 16
    num_heads = 4
    tgt_len = 6
    src_len = 8

    np.random.seed(51)
    # Create input tensors
    pt_query = torch.tensor(
        np.random.rand(tgt_len, batch_size, d_model), dtype=torch.float32
    )
    pt_key = torch.tensor(
        np.random.rand(src_len, batch_size, d_model), dtype=torch.float32
    )
    pt_value = torch.tensor(
        np.random.rand(src_len, batch_size, d_model), dtype=torch.float32
    )

    # Convert to JAX format (batch first)
    jax_query = jnp.transpose(jnp.array(pt_query.numpy()), (1, 0, 2))
    jax_key = jnp.transpose(jnp.array(pt_key.numpy()), (1, 0, 2))
    jax_value = jnp.transpose(jnp.array(pt_value.numpy()), (1, 0, 2))

    # Create weights and biases
    pt_in_proj_weight = torch.tensor(
        np.random.rand(3 * d_model, d_model), dtype=torch.float32
    )
    pt_in_proj_bias = torch.tensor(np.random.rand(3 * d_model), dtype=torch.float32)
    pt_out_proj_weight = torch.tensor(
        np.random.rand(d_model, d_model), dtype=torch.float32
    )
    pt_out_proj_bias = torch.tensor(np.random.rand(d_model), dtype=torch.float32)

    # Create bias_k and bias_v
    pt_bias_k = torch.tensor(np.random.rand(1, 1, d_model), dtype=torch.float32)
    pt_bias_v = torch.tensor(np.random.rand(1, 1, d_model), dtype=torch.float32)

    jax_in_proj_weight = jnp.array(pt_in_proj_weight.numpy())
    jax_in_proj_bias = jnp.array(pt_in_proj_bias.numpy())
    jax_out_proj_weight = jnp.array(pt_out_proj_weight.numpy())
    jax_out_proj_bias = jnp.array(pt_out_proj_bias.numpy())
    jax_bias_k = jnp.array(pt_bias_k.numpy()).reshape(1, d_model)
    jax_bias_v = jnp.array(pt_bias_v.numpy()).reshape(1, d_model)

    # Run PyTorch implementation
    pt_output, pt_attn_weights = F.multi_head_attention_forward(
        query=pt_query,
        key=pt_key,
        value=pt_value,
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=pt_in_proj_weight,
        in_proj_bias=pt_in_proj_bias,
        bias_k=pt_bias_k,
        bias_v=pt_bias_v,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=pt_out_proj_weight,
        out_proj_bias=pt_out_proj_bias,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
    )

    pt_output_numpy = pt_output.detach().numpy()

    # Setup JAX implementation
    partial_mha = ft.partial(
        multi_head_attention_forward,
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=jax_in_proj_weight,
        in_proj_bias=jax_in_proj_bias,
        bias_k=jax_bias_k,
        bias_v=jax_bias_v,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=jax_out_proj_weight,
        out_proj_bias=jax_out_proj_bias,
        inference=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=False,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        static_k=None,
        static_v=None,
        average_attn_weights=True,
        is_causal=False,
        dropout_key=None,
    )

    # Run JAX implementation with vmap
    jax_output, jax_attn_weights = eqx.filter_vmap(partial_mha, out_axes=(1, 0))(
        jax_query,
        jax_key,
        jax_value,
    )

    # Compare outputs
    output_close = np.allclose(pt_output_numpy, jax_output, rtol=1e-5, atol=1e-5)
    assert output_close

    # Check that attention weights include the additional bias position
    assert pt_attn_weights is not None
    assert pt_attn_weights.shape[-1] == src_len + 1  # +1 for bias_k
    assert jax_attn_weights.shape[-1] == src_len + 1

    # Compare attention weights
    assert pt_attn_weights is not None
    attn_close = np.allclose(
        pt_attn_weights.numpy(), jax_attn_weights, rtol=1e-5, atol=1e-5
    )
    assert attn_close


@pytest.mark.parametrize(
    "B, H, W, C, window_size, num_heads, shift_size, attention_dropout, dropout, qkv_bias_flag, proj_bias_flag, logit_scale_flag, inference",  # noqa
    [
        (12, 56, 56, 96, [7, 7], 3, [0, 0], 0.0, 0.0, True, True, True, True),
        # Case 1: Swin-T Stage 1, No Shift
        (12, 56, 56, 96, [7, 7], 3, [0, 0], 0.0, 0.0, True, True, False, True),
        # Case 2: Swin-T Stage 1, Shift
        (12, 56, 56, 96, [7, 7], 3, [3, 3], 0.0, 0.0, True, True, False, True),
        (1, 56, 56, 96, [7, 7], 3, [3, 3], 0.0, 0.0, True, True, False, True),
        # Case 3: Swin-T Stage 2, No Shift
        (12, 28, 28, 192, [7, 7], 6, [0, 0], 0.0, 0.0, True, True, False, True),
        # Case 4: Swin-T Stage 2, Shift
        (12, 28, 28, 192, [7, 7], 6, [3, 3], 0.0, 0.0, True, True, False, True),
        # Case 5: Swin-T Stage 3, No Shift
        (12, 14, 14, 384, [7, 7], 12, [0, 0], 0.0, 0.0, True, True, False, True),
        # Case 6: Swin-T Stage 3, Shift
        (12, 14, 14, 384, [7, 7], 12, [3, 3], 0.0, 0.0, True, True, False, True),
        # Case 7: Swin-T Stage 4, No Shift
        (12, 7, 7, 768, [7, 7], 24, [0, 0], 0.0, 0.0, True, True, False, True),
        # Case 8: Swin-T Stage 4, Shift
        (12, 7, 7, 768, [7, 7], 24, [3, 3], 0.0, 0.0, True, True, False, True),
        # Case 9: NAT Stage 1, No Shift (with logit_scale)
        (12, 64, 64, 96, [8, 8], 3, [0, 0], 0.0, 0.0, True, True, True, True),
        # Case 10: NAT Stage 1, Shift (with logit_scale)
        (12, 64, 64, 96, [8, 8], 3, [4, 4], 0.0, 0.0, True, True, True, True),
        # Case 11: NAT Stage 2, No Shift
        (12, 32, 32, 192, [8, 8], 6, [0, 0], 0.0, 0.0, True, True, True, True),
        # Case 12: NAT Stage 2, Shift
        (12, 32, 32, 192, [8, 8], 6, [4, 4], 0.0, 0.0, True, True, True, True),
        # Case 13: NAT Stage 3, No Shift
        (12, 16, 16, 384, [8, 8], 12, [0, 0], 0.0, 0.0, True, True, True, True),
        # Case 14: NAT Stage 3, Shift
        (12, 16, 16, 384, [8, 8], 12, [4, 4], 0.0, 0.0, True, True, True, True),
        # Case 15: NAT Stage 4, No Shift
        (12, 8, 8, 768, [8, 8], 24, [0, 0], 0.0, 0.0, True, True, True, True),
        # Case 16: NAT Stage 4, Shift
        (12, 8, 8, 768, [8, 8], 24, [4, 4], 0.0, 0.0, True, True, True, True),
    ],
)
def test_shifted_window_attention(
    B,
    H,
    W,
    C,
    window_size,
    num_heads,
    shift_size,
    attention_dropout,
    dropout,
    qkv_bias_flag,
    proj_bias_flag,
    logit_scale_flag,
    inference,
):
    """Test shifted window attention with various configurations."""
    np.random.seed(42)

    # Calculate derived dimensions based on PyTorch conventions and logs
    # qkv_weight maps C -> 3*C, shape (3*C, C)
    qkv_weight_shape = (3 * C, C)
    # proj_weight maps C -> C, shape (C, C)
    proj_weight_shape = (C, C)
    # rpb shape (1, num_heads, win_h*win_h, win_w*win_w) - batch=1 from logs
    rpb_shape = (
        1,
        num_heads,
        window_size[0] * window_size[0],
        window_size[1] * window_size[1],
    )
    # qkv_bias shape (3*C,)
    qkv_bias_shape = (3 * C,) if qkv_bias_flag else None
    # proj_bias shape (C,)
    proj_bias_shape = (C,) if proj_bias_flag else None
    # logit_scale shape (num_heads, 1, 1)
    logit_scale_shape = (num_heads, 1, 1) if logit_scale_flag else None

    # Generate random inputs based on parameters
    input_data = np.random.uniform(low=-1, high=1, size=(B, H, W, C))
    qkv_weight = np.random.uniform(low=-1, high=1, size=qkv_weight_shape)
    proj_weight = np.random.uniform(low=-1, high=1, size=proj_weight_shape)
    relative_position_bias = np.random.uniform(low=-1, high=1, size=rpb_shape)

    qkv_bias = (
        np.random.uniform(low=-1, high=1, size=qkv_bias_shape)
        if qkv_bias_flag
        else None
    )
    proj_bias = (
        np.random.uniform(low=-1, high=1, size=proj_bias_shape)
        if proj_bias_flag
        else None
    )
    logit_scale = (
        np.random.uniform(low=-1, high=1, size=logit_scale_shape)
        if logit_scale_flag
        else None
    )

    # Convert numpy arrays to torch tensors
    pt_input = torch.tensor(input_data, dtype=torch.float32)
    pt_qkv_weight = torch.tensor(qkv_weight, dtype=torch.float32)
    pt_proj_weight = torch.tensor(proj_weight, dtype=torch.float32)
    pt_relative_position_bias = torch.tensor(
        relative_position_bias, dtype=torch.float32
    )
    pt_qkv_bias = (
        torch.tensor(qkv_bias, dtype=torch.float32) if qkv_bias is not None else None
    )
    pt_proj_bias = (
        torch.tensor(proj_bias, dtype=torch.float32) if proj_bias is not None else None
    )
    pt_logit_scale = (
        torch.tensor(logit_scale, dtype=torch.float32)
        if logit_scale is not None
        else None
    )

    # Call the PyTorch shifted_window_attention function
    print("========= TORCH =========")
    pt_output = torch_shifted_window_attention(
        input=pt_input,
        qkv_weight=pt_qkv_weight,
        proj_weight=pt_proj_weight,
        relative_position_bias=pt_relative_position_bias,
        window_size=window_size,
        num_heads=num_heads,
        shift_size=shift_size,
        attention_dropout=attention_dropout,
        dropout=dropout,
        qkv_bias=pt_qkv_bias,
        proj_bias=pt_proj_bias,
        logit_scale=pt_logit_scale,
        training=not inference,
    )
    print("========= TORCH =========")
    assert pt_output is not None
    assert pt_output.shape == tuple([B, H, W, C])

    # Placeholder for JAX implementation and comparison
    # jax_input = jnp.array(input_data)
    # ... JAX conversion of weights/biases ...

    pt_shifted_window_attention = ft.partial(
        shifted_window_attention,
        qkv_weight=jnp.array(qkv_weight),
        proj_weight=jnp.array(proj_weight),
        relative_position_bias=jnp.array(relative_position_bias),
        window_size=window_size,
        num_heads=num_heads,
        shift_size=shift_size,
        attention_dropout=attention_dropout,
        dropout=dropout,
        qkv_bias=jnp.array(qkv_bias),
        proj_bias=jnp.array(proj_bias),
        logit_scale=jnp.array(logit_scale) if logit_scale is not None else None,
        inference=inference,
        key=jax.random.key(42),
    )

    print("========= JAX =========")
    jax_output = eqx.filter_vmap(pt_shifted_window_attention)(jnp.array(input_data))
    print("========= JAX =========")

    assert jax_output.shape == pt_output.shape

    print(jnp.mean(jax_output), torch.mean(pt_output))

    np.testing.assert_allclose(jax_output, pt_output.detach().numpy(), atol=1.5e-2)

    # test jitting
    pt_shifted_window_attention = eqx.filter_jit(pt_shifted_window_attention)

    jax_output = eqx.filter_vmap(pt_shifted_window_attention)(jnp.array(input_data))

    np.testing.assert_allclose(jax_output, pt_output.detach().numpy(), atol=1.5e-2)
