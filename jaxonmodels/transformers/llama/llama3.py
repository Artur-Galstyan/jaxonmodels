from functools import partial
from typing import Literal, Optional, Union

import equinox as eqx
import jax
from jaxtyping import Array, Bool, Int, PRNGKeyArray

from jaxonmodels.layers.mha import MultiheadAttention
from jaxonmodels.layers.rope_embeddings import process_heads
from jaxonmodels.transformers.llama.model_args import get_dtype, LLaMAModelArgs


class FFN(eqx.Module):
    model_args: LLaMAModelArgs = eqx.field(static=True)
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(self, model_args: LLaMAModelArgs, *, key: PRNGKeyArray):
        self.model_args = model_args
        dtype = get_dtype(model_args.precision)
        w1_key, w2_key, w3_key = jax.random.split(key, 3)
        hidden_dim = 4 * model_args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if model_args.ffn_dim_multiplier is not None:
            hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
        mul_of = model_args.multiple_of
        hidden_dim = mul_of * ((hidden_dim + mul_of - 1) // mul_of)

        self.w1 = eqx.nn.Linear(
            model_args.dim, hidden_dim, use_bias=False, key=w1_key, dtype=dtype
        )
        self.w2 = eqx.nn.Linear(
            hidden_dim, model_args.dim, use_bias=False, key=w2_key, dtype=dtype
        )
        self.w3 = eqx.nn.Linear(
            model_args.dim, hidden_dim, use_bias=False, key=w3_key, dtype=dtype
        )

    def __call__(self, x: Array) -> Array:
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


class LLaMABlock(eqx.Module):
    model_args: LLaMAModelArgs = eqx.field(static=True)
    attention: MultiheadAttention
    rope: eqx.nn.RotaryPositionalEmbedding
    attention_norm: eqx.nn.RMSNorm
    ffn_norm: eqx.nn.RMSNorm
    ffn: FFN

    def __init__(self, model_args: LLaMAModelArgs, *, key: PRNGKeyArray):
        self.model_args = model_args
        attention_key, ffn_key = jax.random.split(key, 2)
        dtype = get_dtype(model_args.precision)
        self.attention = MultiheadAttention(
            num_heads=model_args.n_heads,
            kv_multihead_dim=model_args.n_kv_heads,
            query_size=model_args.head_dim * model_args.n_heads,
            kv_interpolation_mode="repeat",
            state_length=model_args.max_seq_len,
            key=attention_key,
            dtype=dtype,
            use_query_bias=False,
            use_key_bias=False,
            use_value_bias=False,
            use_output_bias=False,
        )

        self.rope = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=model_args.head_dim, theta=model_args.rope_theta
        )
        self.attention_norm = eqx.nn.RMSNorm(
            shape=model_args.dim, eps=model_args.norm_eps, dtype=dtype, use_bias=False
        )
        self.ffn_norm = eqx.nn.RMSNorm(
            shape=model_args.dim, eps=model_args.norm_eps, dtype=dtype, use_bias=False
        )
        self.ffn = FFN(model_args, key=ffn_key)

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        mask: Union[
            None,
            Bool[Array, "q_seq kv_seq"],
            Bool[Array, "num_heads q_seq kv_seq"],
            Literal["causal"],
        ] = None,
        *,
        key: Optional[PRNGKeyArray],
        inference: bool = False,
    ) -> tuple[Array, eqx.nn.State]:
        process_heads_fn = partial(process_heads, self.rope)

        h, state = self.attention(
            query=jax.vmap(self.attention_norm)(x),
            key_=jax.vmap(self.attention_norm)(x),
            value=jax.vmap(self.attention_norm)(x),
            mask=mask,
            state=state,
            key=key,
            process_heads=process_heads_fn,  # type: ignore
            inference=inference,
        )
        h = x + h
        normed_h = eqx.filter_vmap(self.ffn_norm)(h)
        out = h + eqx.filter_vmap(self.ffn)(normed_h)
        return out, state


class LLaMA(eqx.Module):
    model_args: LLaMAModelArgs = eqx.field(static=True)
    tok_embeddings: eqx.nn.Embedding
    norm: eqx.nn.RMSNorm
    layers: list[LLaMABlock]
    output: eqx.nn.Linear

    def __init__(self, model_args: LLaMAModelArgs, *, key: PRNGKeyArray):
        self.model_args = model_args
        dtype = get_dtype(model_args.precision)
        tok_embeddings_key, output_key, *block_keys = jax.random.split(
            key, 2 + model_args.n_layers
        )

        self.tok_embeddings = eqx.nn.Embedding(
            model_args.vocab_size, model_args.dim, key=tok_embeddings_key, dtype=dtype
        )
        self.norm = eqx.nn.RMSNorm(
            shape=model_args.dim, eps=model_args.norm_eps, dtype=dtype, use_bias=False
        )
        self.layers = [
            LLaMABlock(model_args, key=block_key) for block_key in block_keys
        ]
        self.output = eqx.nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            key=output_key,
            dtype=dtype,
            use_bias=False,
        )

    def __call__(
        self,
        x: Int[Array, " seq_length"],
        state: eqx.nn.State,
        mask: Union[
            None,
            Bool[Array, "q_seq kv_seq"],
            Bool[Array, "num_heads q_seq kv_seq"],
            Literal["causal"],
        ] = None,
        inference: bool = False,
        *,
        key: Optional[PRNGKeyArray],
    ):
        h = jax.vmap(self.tok_embeddings)(x)
        for layer in self.layers:
            h, state = layer(h, state=state, mask=mask, inference=inference, key=key)
        h = eqx.filter_vmap(self.norm)(h)
        return jax.vmap(self.output)(h), state
