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
from jaxonmodels.layers.normalization import LayerNorm


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

    inference: bool

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
        inference: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ) -> None:
        self.inference = inference
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
                inference=self.inference,
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
                inference=self.inference,
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


# For ESM
class GeometricReasoningOriginalImpl(eqx.Module):
    c_s: int = eqx.field(static=True)
    v_heads: int = eqx.field(static=True)
    num_vector_messages: int = eqx.field(static=True)
    mask_and_zero_frameless: bool = eqx.field(static=True)

    distance_scale_per_head: Array
    rotation_scale_per_head: Array

    def __init__(
        self,
        c_s: int,
        v_heads: int,
        num_vector_messages: int = 1,
        mask_and_zero_frameless: bool = True,
        divide_residual_by_depth: bool = False,
        use_bias: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        """Approximate implementation:

        ATTN(A, v) := (softmax_j A_ij) v_j
        make_rot_vectors(x) := R(i->g) Linear(x).reshape(..., 3)
        make_vectors(x) := T(i->g) Linear(x).reshape(..., 3)

        v <- make_rot_vectors(x)
        q_dir, k_dir <- make_rot_vectors(x)
        q_dist, k_dist <- make_vectors(x)

        A_ij       <- dot(q_dir_i, k_dir_j) -||q_dist_i - k_dist_j||^2
        x          <- x + Linear(T(g->i) ATTN(A, v))
        """
        self.c_s = c_s
        self.v_heads = v_heads
        self.num_vector_messages = num_vector_messages
        self.mask_and_zero_frameless = mask_and_zero_frameless

        self.s_norm = LayerNorm(c_s, use_bias=use_bias)
        dim_proj = (
            4 * self.v_heads * 3 + self.v_heads * 3 * self.num_vector_messages
        )  # 2 x (q, k) * number of heads * (x, y, z) + number of heads * number of vector messages * (x, y, z) # noqa
        key, *subkeys = jax.random.split(key, 3)
        self.proj = eqx.nn.Linear(c_s, dim_proj, use_bias=use_bias, key=subkeys[0])
        channels_out = self.v_heads * 3 * self.num_vector_messages
        self.out_proj = eqx.nn.Linear(
            channels_out, c_s, use_bias=use_bias, key=subkeys[1]
        )

        # The basic idea is for some attention heads to pay more or less attention to
        # rotation versus distance, as well as to control the sharpness of the softmax
        # (i.e., should this head only attend to those residues
        # very nearby or should there be shallower dropoff in attention weight?)
        self.distance_scale_per_head = jnp.zeros((self.v_heads))
        self.rotation_scale_per_head = jnp.zeros((self.v_heads))

    def __call__(self, s, affine, affine_mask, sequence_id, chain_id):
        if sequence_id is None:
            sequence_id = jnp.zeros_like(s[..., 0], dtype=jnp.int64)
        attn_bias = jnp.expand_dims(sequence_id, (-1)) == jnp.expand_dims(
            sequence_id, -2
        )
        attn_bias = jnp.expand_dims(attn_bias, 1)

        attn_bias = attn_bias.at[~affine_mask[:, None, None, :]].set(
            jnp.finfo(attn_bias.dtype).min
        )
        chain_id_mask = jnp.expand_dims(chain_id, 1) != jnp.expand_dims(chain_id, 2)

        mask = jnp.expand_dims(chain_id_mask, axis=1)
        attn_bias = attn_bias.at[mask].set(jnp.finfo(s.dtype).min)

        ns = self.s_norm(s)
        vec_rot, vec_dist = jnp.split(
            self.proj(ns),
            [
                self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages,
                self.v_heads * 2 * 3,
            ],
            axis=-1,
        )

        # Rotate the queries and keys for the rotation term. We also rotate the values.
        # NOTE(zeming, thayes): Values are only rotated, not translated. We may wish to change
        # this in the future.
        query_rot, key_rot, value = (
            affine.rot[..., None]
            .apply(rearrange(vec_rot, "... (h c) -> ... h c", c=3))
            .split(
                [self.v_heads, self.v_heads, self.v_heads * self.num_vector_messages],
                dim=-2,
            )
        )

        # Rotate and translate the queries and keys for the distance term
        # NOTE(thayes): a simple speedup would be to apply all rotations together, then
        # separately apply the translations.
        query_dist, key_dist = (
            affine[..., None]
            .apply(rearrange(vec_dist, "... (h c) -> ... h c", c=3))
            .chunk(2, dim=-2)
        )

        query_dist = rearrange(query_dist, "b s h d -> b h s 1 d")
        key_dist = rearrange(key_dist, "b s h d -> b h 1 s d")
        query_rot = rearrange(query_rot, "b s h d -> b h s d")
        key_rot = rearrange(key_rot, "b s h d -> b h d s")
        value = rearrange(
            value, "b s (h m) d -> b h s (m d)", m=self.num_vector_messages
        )

        distance_term = (query_dist - key_dist).norm(dim=-1) / sqrt(3)
        rotation_term = query_rot.matmul(key_rot) / sqrt(3)
        distance_term_weight = rearrange(
            F.softplus(self.distance_scale_per_head), "h -> h 1 1"
        )
        rotation_term_weight = rearrange(
            F.softplus(self.rotation_scale_per_head), "h -> h 1 1"
        )

        attn_weight = (
            rotation_term * rotation_term_weight - distance_term * distance_term_weight
        )

        if attn_bias is not None:
            # we can re-use the attention bias from the transformer layers
            # NOTE(thayes): This attention bias is expected to handle two things:
            # 1. Masking attention on padding tokens
            # 2. Masking cross sequence attention in the case of bin packing
            s_q = attn_weight.size(2)
            s_k = attn_weight.size(3)
            _s_q = max(0, attn_bias.size(2) - s_q)
            _s_k = max(0, attn_bias.size(3) - s_k)
            attn_bias = attn_bias[:, :, _s_q:, _s_k:]
            attn_weight = attn_weight + attn_bias

        attn_weight = torch.softmax(attn_weight, dim=-1)

        attn_out = attn_weight.matmul(value)

        attn_out = (
            affine.rot[..., None]
            .invert()
            .apply(
                rearrange(
                    attn_out, "b h s (m d) -> b s (h m) d", m=self.num_vector_messages
                )
            )
        )

        attn_out = rearrange(
            attn_out, "b s (h m) d -> b s (h m d)", m=self.num_vector_messages
        )
        if self.mask_and_zero_frameless:
            attn_out = attn_out.masked_fill(~affine_mask[..., None], 0.0)
        s = self.out_proj(attn_out)

        return s
