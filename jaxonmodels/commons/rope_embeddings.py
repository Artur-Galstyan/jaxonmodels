from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray


internal_rope_embedding_cache: dict[int, Array] = {}


class RotaryPositionalEmbedding(eqx.Module):
    embedding_size: int = eqx.field(static=True)

    def __check_init__(self):
        if self.embedding_size < 0:
            raise ValueError("`embedding_size` must not be negative.")
        if (self.embedding_size % 2) != 0:
            raise ValueError("`embedding_size` must be even.")

    @staticmethod
    def rotate_half(x: Float[Array, "seq_length embedding_size"]):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([-x[..., d_2:], x[..., :d_2]], axis=-1)

    @staticmethod
    def precompute_freqs_cis(
        embedding_size: int, end: int, theta: float = 10000.0
    ) -> Complex[Array, "end half_of_embedding_size"]:
        freqs = 1.0 / (
            theta
            ** (jnp.arange(0.0, embedding_size, 2)[jnp.newaxis, :] / embedding_size)
        )

        t = jnp.arange(float(end))
        freqs_outer = jnp.outer(t, freqs)
        with jax.numpy_dtype_promotion("standard"):
            freqs_cis = jnp.cos(freqs_outer) + jnp.sin(freqs_outer) * 1j

        return freqs_cis

    def __call__(
        self,
        x: Float[Array, "seq_length embedding_size"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq_length embedding_size"]:
        """**Arguments:**

        - `x`: A JAX array of shape `(seq_length, embedding_size)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(seq_length, embedding_size)`, with the rotary positional
        encoding applied to the input.
        """

        seq_len, embedding_size = x.shape
        if embedding_size != self.embedding_size:
            raise ValueError(
                f"x.shape[-1] must match self.embedding_size, "
                f"but {x.shape[-1]} != {self.embedding_size}"
            )

        with jax.ensure_compile_time_eval():
            if embedding_size in internal_rope_embedding_cache:
                freqs_cis = internal_rope_embedding_cache[embedding_size]
                freqs_cis_seq_len, _ = freqs_cis.shape
                if seq_len > freqs_cis_seq_len:
                    freqs_cis = self.precompute_freqs_cis(embedding_size, seq_len)
                    internal_rope_embedding_cache[embedding_size] = freqs_cis
                else:
                    freqs_cis = freqs_cis[:seq_len]
            else:
                freqs_cis = self.precompute_freqs_cis(embedding_size, seq_len)
                internal_rope_embedding_cache[embedding_size] = freqs_cis

        freqs_real = jnp.tile(freqs_cis.real, (1, 2))
        freqs_imag = jnp.tile(freqs_cis.imag, (1, 2))

        rotate_x = self.rotate_half(x)
        x_rope = (x * freqs_real) + (rotate_x * freqs_imag)
        return x_rope


def process_heads(
    rope_embeddings: RotaryPositionalEmbedding,
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
