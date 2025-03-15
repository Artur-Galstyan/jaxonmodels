import functools as ft

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from jaxonmodels.functions.functions import multi_head_attention_forward


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
        training=False,
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
