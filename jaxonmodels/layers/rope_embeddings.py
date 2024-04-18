import equinox as eqx
import jax
from jaxtyping import Array, Float


def process_heads(
    rope_embeddings: eqx.nn.RotaryPositionalEmbedding,
    query_heads: Float[Array, "seq_length num_heads qk_size"],
    key_heads: Float[Array, "seq_length num_heads qk_size"],
    value_heads: Float[Array, "seq_length num_heads vo_size"],
) -> tuple[
    Float[Array, "seq_length num_heads qk_size"],
    Float[Array, "seq_length num_heads qk_size"],
    Float[Array, "seq_length num_heads vo_size"],
]:
    query_heads = jax.vmap(rope_embeddings, in_axes=1, out_axes=1)(query_heads)
    key_heads = jax.vmap(rope_embeddings, in_axes=1, out_axes=1)(key_heads)

    return query_heads, key_heads, value_heads
