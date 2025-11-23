import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxonlayers.layers import MultiheadAttention
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions.utils import default_floating_dtype


class ResidualAttentionBlock(eqx.Module):
    attn: MultiheadAttention
    ln_1: eqx.nn.LayerNorm
    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    ln_2: eqx.nn.LayerNorm

    def __init__(self, d_model: int, n_head: int, *, key: PRNGKeyArray, dtype: Any):
        key, *subkeys = jax.random.split(key, 5)
        self.attn = MultiheadAttention(d_model, n_head, key=subkeys[0], dtype=dtype)
        self.ln_1 = eqx.nn.LayerNorm(d_model, dtype=dtype)
        self.c_fc = eqx.nn.Linear(d_model, d_model * 4, key=subkeys[1], dtype=dtype)
        self.c_proj = eqx.nn.Linear(d_model * 4, d_model, key=subkeys[2], dtype=dtype)
        self.ln_2 = eqx.nn.LayerNorm(d_model, dtype=dtype)

    def _attention(self, x: Array, attn_mask: Array | None = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def __call__(self, x: Array, attn_mask: Array | None = None):
        ln1_out = eqx.filter_vmap(self.ln_1)(x)
        attention_output = self._attention(ln1_out, attn_mask)
        x = x + attention_output
        ln_2_output = eqx.filter_vmap(self.ln_2)(x)
        c_fc_out = eqx.filter_vmap(self.c_fc)(ln_2_output)
        x_gelu = jax.nn.gelu(c_fc_out)
        c_proj_out = eqx.filter_vmap(self.c_proj)(x_gelu)
        x = x + c_proj_out  # Add to x, not to ln_2_output
        return x


class Transformer(eqx.Module):
    resblocks: list[ResidualAttentionBlock]

    def __init__(
        self, width: int, layers: int, heads: int, *, key: PRNGKeyArray, dtype: Any
    ):
        key, *subkeys = jax.random.split(key, layers + 1)
        self.resblocks = [
            ResidualAttentionBlock(width, heads, key=subkeys[i], dtype=dtype)
            for i in range(layers)
        ]

    def __call__(
        self, x: Float[Array, "seq_len embed_dim"], attn_mask: Array | None = None
    ):
        for resblock in self.resblocks:
            x = resblock(x, attn_mask=attn_mask)

        return x


class VisionTransformer(eqx.Module):
    class_embedding: Array
    positional_embedding: Array
    proj: Array
    conv1: eqx.nn.Conv2d
    ln_pre: eqx.nn.LayerNorm
    transformer: Transformer
    ln_post: eqx.nn.LayerNorm

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        key, *subkeys = jax.random.split(key, 6)
        self.conv1 = eqx.nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            use_bias=False,
            key=subkeys[0],
        )

        scale = width**-0.5
        self.class_embedding = (
            jax.random.normal(subkeys[1], (width,), dtype=dtype) * scale
        )
        self.positional_embedding = scale * jax.random.normal(
            subkeys[2], ((input_resolution // patch_size) ** 2 + 1, width), dtype=dtype
        )

        self.ln_pre = eqx.nn.LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads, key=subkeys[3], dtype=dtype
        )

        self.ln_post = eqx.nn.LayerNorm(width, dtype=dtype)
        self.proj = scale * jax.random.normal(
            subkeys[4], (width, output_dim), dtype=dtype
        )

    def __call__(
        self,
        x: Float[Array, "c h w"],
        state: eqx.nn.State | None = None,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], -1)
        x = jnp.transpose(x)

        x = jnp.concatenate(
            [
                self.class_embedding.reshape(1, -1),
                x,
            ],
        )
        x = x + self.positional_embedding
        x = eqx.filter_vmap(self.ln_pre)(x)
        x = self.transformer(x)
        x = self.ln_post(x[0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, state
