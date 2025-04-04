import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from jaxonmodels.functions import (
    stochastic_depth,
)


class StochasticDepth(eqx.Module):
    p: float = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    inference: bool

    def __init__(self, p: float, mode: str, inference: bool = False) -> None:
        self.p = p
        self.mode = mode
        self.inference = inference

    def __call__(self, input: Array, key: PRNGKeyArray) -> Array:
        return stochastic_depth(input, self.p, self.mode, self.inference, key)
