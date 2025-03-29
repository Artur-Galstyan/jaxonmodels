import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array, PRNGKeyArray

from jaxonmodels.functions import (
    canonical_mask,
    multi_head_attention_forward,
)
from jaxonmodels.functions.utils import default_floating_dtype


class MultiheadAttention(eqx.Module):
    q_proj_weight: Array | None
    k_proj_weight: Array | None
    v_proj_weight: Array | None

    in_proj_weight: Array | None

    in_proj_bias: Array | None

    out_proj: eqx.nn.Linear

    bias_k: Array | None
    bias_v: Array | None

    embed_dim: int = eqx.field(static=True)
    kdim: int = eqx.field(static=True)
    vdim: int = eqx.field(static=True)
    _qkv_same_embed_dim: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    add_zero_attn: bool = eqx.field(static=True)

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ) -> None:
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        uniform_initializer = jax.nn.initializers.uniform(dtype=dtype)

        if not self._qkv_same_embed_dim:
            key, *subkeys = jax.random.split(key, 4)
            self.q_proj_weight = uniform_initializer(
                key=subkeys[0], shape=(embed_dim, embed_dim)
            )
            self.k_proj_weight = uniform_initializer(
                key=subkeys[1], shape=(embed_dim, self.kdim)
            )
            self.v_proj_weight = uniform_initializer(
                key=subkeys[2], shape=(embed_dim, self.vdim)
            )
            self.in_proj_weight = None
        else:
            key, subkey = jax.random.split(key)
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
            self.in_proj_weight = uniform_initializer(
                key=subkey, shape=(3 * embed_dim, embed_dim)
            )

        if bias:
            self.in_proj_bias = jnp.empty((3 * embed_dim), dtype=dtype)
        else:
            self.in_proj_bias = None
        key, subkey = jax.random.split(key)
        out_proj = eqx.nn.Linear(
            embed_dim, embed_dim, use_bias=bias, key=subkey, dtype=dtype
        )
        if bias:
            assert out_proj.bias is not None
            new_bias = jnp.zeros_like(out_proj.bias, dtype=dtype)
            where = lambda l: l.bias
            self.out_proj = eqx.tree_at(where, out_proj, new_bias)
        else:
            self.out_proj = out_proj

        if add_bias_kv:
            normal_initializer = jax.nn.initializers.normal(dtype=dtype)
            key, *subkeys = jax.random.split(key, 3)
            self.bias_k = normal_initializer(key=subkeys[0], shape=(1, embed_dim))
            self.bias_v = normal_initializer(key=subkeys[0], shape=(1, embed_dim))
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        key_padding_mask: Array | None = None,
        need_weights: bool = True,
        attn_mask: Array | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        inference: bool = False,
    ) -> tuple[Array, Array | None]:
        key_padding_mask = canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                inference=inference,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                inference=inference,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        return attn_output, attn_output_weights


class SqueezeExcitation(eqx.Module):
    avgpool: eqx.nn.AdaptiveAvgPool2d
    fc1: eqx.nn.Conv2d
    fc2: eqx.nn.Conv2d

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ) -> None:
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.avgpool = eqx.nn.AdaptiveAvgPool2d(1)
        key, subkey = jax.random.split(key)
        self.fc1 = eqx.nn.Conv2d(
            input_channels, squeeze_channels, 1, key=key, dtype=dtype
        )
        self.fc2 = eqx.nn.Conv2d(
            squeeze_channels, input_channels, 1, key=subkey, dtype=dtype
        )

    def __call__(
        self,
        x: Array,
        activation: Callable[..., Array] = jax.nn.relu,
        scale_activation: Callable[..., Array] = jax.nn.sigmoid,
    ) -> Array:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = activation(scale)
        scale = self.fc2(scale)
        scale = scale_activation(scale)
        return scale * x
