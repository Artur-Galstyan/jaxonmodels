import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


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


class SelectiveStateSpaceModel(eqx.Module):
    input_proj: eqx.nn.Linear
    delta_proj: eqx.nn.Linear
    A_log: Float[Array, "d_inner d_state"]
    D: Float[Array, " d_inner"]

    d_inner: int = eqx.field(static=True)
    dt_rank: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)

    def __init__(
        self,
        d_inner: int,
        dt_rank: int,
        d_state: int,
        use_input_proj_bias: bool = False,
        use_delta_proj_bias: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        (
            key,
            input_proj_key,
            delta_proj_key,
        ) = jax.random.split(key, 3)
        self.input_proj = eqx.nn.Linear(
            d_inner,
            dt_rank + d_state * 2,
            use_bias=use_input_proj_bias,
            key=input_proj_key,
        )

        self.delta_proj = eqx.nn.Linear(
            dt_rank, d_inner, use_bias=use_delta_proj_bias, key=delta_proj_key
        )
        A = jnp.repeat(jnp.arange(1, d_state + 1), d_inner).reshape(d_inner, d_state)
        self.A_log = jnp.log(A)
        self.D = jnp.ones(d_inner)

    def __call__(self, x: Float[Array, "seq_length d_inner"]):
        A = -jnp.exp(self.A_log)
        D = self.D

        delta_b_c = jax.vmap(self.input_proj)(x)

        split_indices = [
            self.dt_rank,
            self.dt_rank + self.d_state,
        ]
        delta, B, C = jnp.split(delta_b_c, split_indices, axis=-1)
        delta = jax.nn.softplus(jax.vmap(self.delta_proj)(delta))

        y = selective_scan(x, delta, A, B, C, D)
        return y
