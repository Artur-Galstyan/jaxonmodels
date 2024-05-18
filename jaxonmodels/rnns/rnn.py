import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class LSTM(eqx.Module):
    cell: eqx.nn.LSTMCell

    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray):
        self.cell = eqx.nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size, key=key
        )

    def __call__(self, xs):
        scan_fn = lambda state, input: (self.cell(input, state), None)
        init_state = (
            jnp.zeros(self.cell.hidden_size),
            jnp.zeros(self.cell.hidden_size),
        )
        final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
        return final_state


class RNN(eqx.Module):
    hidden_size: int = eqx.field(static=True)

    i2h: eqx.nn.Linear
    h2o: eqx.nn.Linear

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, *, key: PRNGKeyArray
    ):
        self.hidden_size = hidden_size
        h_key, o_key = jax.random.split(key)
        self.i2h = eqx.nn.Linear(input_size + hidden_size, hidden_size, key=h_key)
        self.h2o = eqx.nn.Linear(input_size + hidden_size, output_size, key=o_key)

    def __call__(
        self, x: Float[Array, " input_size"], hidden: Float[Array, " hidden_size"]
    ) -> tuple[Float[Array, " output_size"], Float[Array, " hidden_size"]]:
        combined = jnp.concatenate((x, hidden), axis=1).reshape(-1)

        hidden = self.i2h(combined).reshape(1, -1)
        output = self.h2o(combined)

        return output, hidden

    def init_hidden(self) -> Float[Array, "1 hidden_size"]:
        return jnp.zeros((1, self.hidden_size))
