import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from jaxonmodels.functions.functions import (
    stochastic_depth,
)


class StochasticDepth(eqx.Module):
    p: float = eqx.field(static=True)
    mode: str = eqx.field(static=True)

    def __init__(self, p: float, mode: str) -> None:
        self.p = p
        self.mode = mode

    def __call__(self, input: Array, inference: bool, key: PRNGKeyArray) -> Array:
        return stochastic_depth(input, self.p, self.mode, inference, key)
