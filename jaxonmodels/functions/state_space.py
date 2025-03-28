import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def selective_scan(
    x: Float[Array, "seq_length d_inner"],
    delta: Float[Array, "seq_length d_inner"],
    A: Float[Array, "d_inner d_state"],
    B: Float[Array, "seq_length d_state"],
    C: Float[Array, "seq_length d_state"],
    D: Float[Array, " d_inner"],
) -> Float[Array, "seq_length d_inner"]:
    L, d_inner = x.shape
    _, d_state = A.shape
    delta_A = jnp.exp(jnp.einsum("l d,d n -> l d n", delta, A))
    delta_B_u = jnp.einsum("l d,l n,l d -> l d n", delta, B, x)

    x_res = jnp.zeros(shape=(d_inner, d_state))

    def step(x, i):
        x = delta_A[i] * x + delta_B_u[i]

        y = jnp.einsum("d n,n -> d", x, C[i, :])
        return x, y

    _, ys = jax.lax.scan(step, x_res, jnp.arange(L))

    ys = ys + x * D
    return ys
