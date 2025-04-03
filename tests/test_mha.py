import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn as nn

from jaxonmodels.layers import MultiheadAttention


def transfer_weights_to_torch(jax_mha, torch_mha):
    """Transfer weights from JAX MHA to PyTorch MHA."""
    # Handle the case where we use separate projection weights
    if not jax_mha._qkv_same_embed_dim:
        # Convert JAX arrays to numpy, then to torch tensors
        torch_mha.q_proj_weight.data = torch.tensor(
            np.array(jax_mha.q_proj_weight), dtype=torch_mha.q_proj_weight.dtype
        )
        torch_mha.k_proj_weight.data = torch.tensor(
            np.array(jax_mha.k_proj_weight), dtype=torch_mha.k_proj_weight.dtype
        )
        torch_mha.v_proj_weight.data = torch.tensor(
            np.array(jax_mha.v_proj_weight), dtype=torch_mha.v_proj_weight.dtype
        )
    else:
        # Set the in_proj_weight for the case where q, k, v dimensions are the same
        torch_mha.in_proj_weight.data = torch.tensor(
            np.array(jax_mha.in_proj_weight), dtype=torch_mha.in_proj_weight.dtype
        )

    # Set bias if it exists
    if jax_mha.in_proj_bias is not None:
        torch_mha.in_proj_bias.data = torch.tensor(
            np.array(jax_mha.in_proj_bias), dtype=torch_mha.in_proj_bias.dtype
        )

    # Set out projection weights and bias
    torch_mha.out_proj.weight.data = torch.tensor(
        np.array(jax_mha.out_proj.weight), dtype=torch_mha.out_proj.weight.dtype
    )

    if jax_mha.out_proj.bias is not None:
        torch_mha.out_proj.bias.data = torch.tensor(
            np.array(jax_mha.out_proj.bias), dtype=torch_mha.out_proj.bias.dtype
        )

    # Set bias_k and bias_v if they exist
    if jax_mha.bias_k is not None:
        torch_mha.bias_k.data = torch.tensor(
            np.array(jax_mha.bias_k), dtype=torch_mha.bias_k.dtype
        )

    if jax_mha.bias_v is not None:
        torch_mha.bias_v.data = torch.tensor(
            np.array(jax_mha.bias_v), dtype=torch_mha.bias_v.dtype
        )


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim",  # noqa
    [
        (2, 10, 64, 8, 0.0, True, False, False, None, None),  # Standard case
        (2, 8, 64, 8, 0.0, False, False, False, None, None),  # No bias
        (2, 10, 64, 8, 0.0, True, True, False, None, None),  # With bias_kv
        (2, 10, 64, 8, 0.0, True, False, True, None, None),  # With zero_attn
    ],
)
def test_multihead_attention_equivalence(
    batch_size,
    seq_len,
    embed_dim,
    num_heads,
    dropout,
    bias,
    add_bias_kv,
    add_zero_attn,
    kdim,
    vdim,
):
    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    # Create random inputs
    key, subkey = jax.random.split(key)
    jax_query = jax.random.normal(subkey, (batch_size, seq_len, embed_dim))

    key, subkey = jax.random.split(key)
    jax_key = jax.random.normal(
        subkey, (batch_size, seq_len, kdim if kdim is not None else embed_dim)
    )

    key, subkey = jax.random.split(key)
    jax_value = jax.random.normal(
        subkey, (batch_size, seq_len, vdim if vdim is not None else embed_dim)
    )

    # Convert to torch tensors
    torch_query = torch.tensor(np.array(jax_query), dtype=torch.float32)
    torch_key = torch.tensor(np.array(jax_key), dtype=torch.float32)
    torch_value = torch.tensor(np.array(jax_value), dtype=torch.float32)

    # Initialize JAX model
    key, subkey = jax.random.split(key)
    jax_mha = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
        kdim=kdim,
        vdim=vdim,
        key=subkey,
    )

    jax_mha = eqx.nn.inference_mode(jax_mha)

    # Initialize PyTorch model
    torch_mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=bias,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
        kdim=kdim,
        vdim=vdim,
    )

    # Transfer weights from JAX to PyTorch
    transfer_weights_to_torch(jax_mha, torch_mha)

    # Run JAX forward pass (with inference=True to disable dropout for comparison)
    jax_mha_pt = functools.partial(
        jax_mha,
        need_weights=True,
    )
    jax_output, jax_attention = eqx.filter_vmap(jax_mha_pt)(
        jax_query,
        jax_key,
        jax_value,
    )

    # PyTorch expects inputs in the shape (seq_len, batch_size, embed_dim)
    # while JAX has shape (batch_size, seq_len, embed_dim)
    torch_query = torch_query.permute(1, 0, 2)
    torch_key = torch_key.permute(1, 0, 2)
    torch_value = torch_value.permute(1, 0, 2)

    # Run PyTorch forward pass
    with torch.no_grad():
        torch_output, torch_attention = torch_mha(
            query=torch_query,
            key=torch_key,
            value=torch_value,
            need_weights=True,
        )

    # Convert PyTorch output back to the JAX format (batch_size, seq_len, embed_dim)
    torch_output = torch_output.permute(1, 0, 2).detach().numpy()

    # If attn_weights are averaged across heads, we compare directly
    # otherwise we may need to reshape and average for comparison
    if torch_attention is not None:
        torch_attention = torch_attention.detach().numpy()

    # Compare outputs
    assert jax_output.shape == torch_output.shape, "Output shapes don't match"
    np.testing.assert_allclose(
        np.array(jax_output),
        torch_output,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Outputs don't match between JAX and PyTorch",
    )

    # Only compare attention weights if they are both not None
    if jax_attention is not None and torch_attention is not None:
        np.testing.assert_allclose(
            np.array(jax_attention),
            torch_attention,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Attention weights don't match between JAX and PyTorch",
        )


@pytest.mark.parametrize(
    "seq_len, embed_dim, num_heads, is_causal",
    [
        (10, 64, 8, True),  # Causal attention
        (10, 64, 8, False),  # Non-causal attention
    ],
)
def test_causal_attention(seq_len, embed_dim, num_heads, is_causal):
    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    batch_size = 2

    # Create random inputs
    key, subkey = jax.random.split(key)
    jax_query = jax.random.normal(subkey, (batch_size, seq_len, embed_dim))
    jax_key = jax_query  # Same as query for self-attention
    jax_value = jax_query

    # Convert to torch tensors
    torch_query = torch.tensor(np.array(jax_query), dtype=torch.float32)
    torch_key = torch_query  # Same as query for self-attention
    torch_value = torch_query

    # Initialize JAX model
    key, subkey = jax.random.split(key)
    jax_mha = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,  # No dropout for deterministic comparison
        bias=True,
        key=subkey,
    )

    jax_mha = eqx.nn.inference_mode(jax_mha)

    # Initialize PyTorch model
    torch_mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,  # No dropout for deterministic comparison
        bias=True,
    )

    # Transfer weights from JAX to PyTorch
    transfer_weights_to_torch(jax_mha, torch_mha)

    jax_mha_pt = functools.partial(
        jax_mha,
        need_weights=True,
        is_causal=is_causal,
    )
    jax_output, jax_attention = eqx.filter_vmap(jax_mha_pt)(
        jax_query,
        jax_key,
        jax_value,
    )

    # PyTorch expects inputs in the shape (seq_len, batch_size, embed_dim)
    torch_query = torch_query.permute(1, 0, 2)
    torch_key = torch_key.permute(1, 0, 2)
    torch_value = torch_value.permute(1, 0, 2)

    # Create causal mask for PyTorch if needed
    torch_attn_mask = None
    if is_causal:
        # Create a causal mask where upper triangle is -inf
        torch_attn_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1
        )

    # Run PyTorch forward pass
    with torch.no_grad():
        torch_output, torch_attention = torch_mha(
            query=torch_query,
            key=torch_key,
            value=torch_value,
            attn_mask=torch_attn_mask,
            need_weights=True,
        )

    # Convert PyTorch output back to the JAX format
    torch_output = torch_output.permute(1, 0, 2).detach().numpy()

    if torch_attention is not None:
        torch_attention = torch_attention.detach().numpy()

    # Compare outputs
    assert jax_output.shape == torch_output.shape, "Output shapes don't match"
    np.testing.assert_allclose(
        np.array(jax_output),
        torch_output,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Outputs don't match between JAX and PyTorch for causal={is_causal}",
    )


def test_key_padding_mask():
    # Test the effect of key_padding_mask
    batch_size, seq_len, embed_dim, num_heads = 2, 10, 64, 8

    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    # Create random inputs
    key, subkey = jax.random.split(key)
    jax_query = jax.random.normal(subkey, (batch_size, seq_len, embed_dim))
    jax_key = jax_query  # Same for self-attention
    jax_value = jax_query

    # Create padding mask (True means to mask)
    # Let's mask the last 3 positions for each sequence in the batch
    jax_key_padding_mask = jnp.zeros((batch_size, seq_len), dtype=bool)
    jax_key_padding_mask = jax_key_padding_mask.at[:, -3:].set(True)

    torch_key_padding_mask = torch.tensor(np.array(jax_key_padding_mask))

    # Initialize models
    key, subkey = jax.random.split(key)
    jax_mha = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        key=subkey,
    )

    jax_mha = eqx.nn.inference_mode(jax_mha)

    torch_mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
    )

    # Transfer weights
    transfer_weights_to_torch(jax_mha, torch_mha)

    # Run JAX forward pass
    jax_mha_pt = functools.partial(
        jax_mha,
        need_weights=True,
    )
    jax_output, jax_attention = eqx.filter_vmap(jax_mha_pt)(
        jax_query, jax_key, jax_value, jax_key_padding_mask
    )

    # PyTorch expects inputs in different shape
    torch_query = torch.tensor(np.array(jax_query), dtype=torch.float32).permute(
        1, 0, 2
    )
    torch_key = torch.tensor(np.array(jax_key), dtype=torch.float32).permute(1, 0, 2)
    torch_value = torch.tensor(np.array(jax_value), dtype=torch.float32).permute(
        1, 0, 2
    )

    # Run PyTorch forward pass
    with torch.no_grad():
        torch_output, torch_attention = torch_mha(
            query=torch_query,
            key=torch_key,
            value=torch_value,
            key_padding_mask=torch_key_padding_mask,
            need_weights=True,
        )

    # Convert PyTorch output back to JAX format
    torch_output = torch_output.permute(1, 0, 2).detach().numpy()

    if torch_attention is not None:
        torch_attention = torch_attention.detach().numpy()

    # Compare outputs
    assert jax_output.shape == torch_output.shape, "Output shapes don't match"
    np.testing.assert_allclose(
        np.array(jax_output),
        torch_output,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Outputs don't match between JAX and PyTorch with key_padding_mask",
    )
