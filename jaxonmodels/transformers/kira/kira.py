from functools import partial
from typing import Optional

import equinox as eqx
import jax
from jaxtyping import Array, Int, PRNGKeyArray

from jaxonmodels.layers.mha import MultiheadAttention
from jaxonmodels.transformers.kira.model_args import KiraModelArgs


class Block(eqx.nn.StatefulLayer):
    mha_attention: MultiheadAttention
    rms_norm: eqx.nn.RMSNorm
    feedforward: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    model_args: KiraModelArgs = eqx.field(static=True)

    rope_embeddings: eqx.nn.RotaryPositionalEmbedding

    def __init__(
        self,
        model_args: KiraModelArgs,
        *,
        key,
        **kwargs,
    ):
        self.model_args = model_args
        key, *subkeys = jax.random.split(key, 5)
        self.rope_embeddings = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=model_args.n_embd,
        )

        self.mha_attention = MultiheadAttention(
            num_heads=model_args.num_heads,
            query_size=model_args.n_embd,
            qk_size=model_args.n_embd,
            key_size=model_args.n_embd,
            value_size=model_args.n_embd,
            vo_size=model_args.n_embd,
            output_size=model_args.n_embd,
            query_multihead_dim=model_args.num_query_heads,
            kv_multihead_dim=model_args.num_kv_heads,
            state_length=model_args.max_seq_len,
            kv_interpolation_mode=model_args.kv_interpolation_mode,
            key=subkeys[0],
        )

        self.rms_norm = eqx.nn.RMSNorm(shape=model_args.n_embd)

        self.feedforward = eqx.nn.MLP(
            model_args.n_embd,
            out_size=model_args.n_embd,
            width_size=model_args.width_size,
            depth=model_args.depth,
            key=subkeys[1],
        )

        self.dropout = eqx.nn.Dropout(p=model_args.p)

    def __call__(
        self,
        x: Int[Array, "max_seq_len input_dim"],
        state: eqx.nn.State | None = None,
        mask: str | None = "causal",
        *,
        key: Optional[PRNGKeyArray],
        **kwargs,
    ):
        def process_heads(query_heads, key_heads, value_heads):
            query_heads = jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(
                query_heads
            )
            key_heads = jax.vmap(self.rope_embeddings, in_axes=1, out_axes=1)(key_heads)

            return query_heads, key_heads, value_heads

        mha_partial = partial(
            self.mha_attention,
            process_heads=process_heads,
            query=jax.vmap(self.rms_norm)(x),
            key_=jax.vmap(self.rms_norm)(x),
            value=jax.vmap(self.rms_norm)(x),
            mask=mask,
        )
        if state is not None:
            mha, state = mha_partial(state=state, key=key)
        else:
            mha = mha_partial(key=key)
        x = mha + x
        inference = True if key is None else False
        d_key1 = None
        d_key2 = None
        if not inference and key is not None:
            key, d_key1, d_key2 = jax.random.split(key, 3)
        x = self.dropout(x, key=d_key1, inference=inference)
        ff = jax.vmap(self.feedforward)(jax.vmap(self.rms_norm)(x))
        x = ff + x
        x = self.dropout(x, key=d_key2, inference=inference)
        return x, state


class Kira(eqx.Module):
    model_args: KiraModelArgs = eqx.field(static=True)

    blocks: list[Block]

    input_embedding: eqx.nn.Embedding
    output: eqx.nn.Linear

    rms_norm: eqx.nn.RMSNorm

    def __init__(
        self,
        model_args: KiraModelArgs,
        *,
        key,
        **kwargs,
    ):
        assert model_args.n_layers is not None, "n_layers must be provided"
        self.model_args = model_args
        key, *subkeys = jax.random.split(key, model_args.n_layers + 2)

        self.input_embedding = eqx.nn.Embedding(
            model_args.n_dims, model_args.n_embd, key=subkeys[0]
        )
        self.blocks = [
            Block(
                model_args=model_args,
                key=subkeys[i + 1],
            )
            for i in range(model_args.n_layers)
        ]

        self.rms_norm = eqx.nn.RMSNorm(shape=model_args.n_embd)
        self.output = eqx.nn.Linear(
            model_args.n_embd, model_args.n_dims, key=subkeys[-1]
        )

    def __call__(
        self,
        x: Int[Array, " seq_len"],
        state: eqx.nn.State | None = None,
        mask: str | None = "causal",
        *,
        key: PRNGKeyArray | None = None,
    ):
        x = jax.vmap(self.input_embedding)(x)
        for block in self.blocks:
            x, state = block(x, state=state, mask=mask, key=key)
        x = jax.vmap(self.rms_norm)(x)
        x = jax.vmap(self.output)(x)

        if state is not None:
            return x, state
        else:
            return x
