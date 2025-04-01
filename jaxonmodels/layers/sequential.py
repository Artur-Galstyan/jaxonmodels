import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray


class Sequential(eqx.Module):
    layers: list[eqx.Module]

    def __init__(self, layers: list[eqx.Module]) -> None:
        self.layers = layers

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
        *args,
        **kwargs,
    ) -> tuple[Array, eqx.nn.State]:
        key, *subkeys = jax.random.split(key, len(self.layers) + 1)
        for i, layer in enumerate(self.layers):
            x, state = layer(x, state, subkeys[i], *args, **kwargs)  # pyright: ignore
        return x, state
