import abc

import equinox as eqx
from jaxtyping import Array


class AbstractNorm(eqx.Module):
    @abc.abstractmethod
    def __call__(self, x: Array, *_, **__) -> Array: ...


class AbstractNormStateful(eqx.nn.StatefulLayer):
    @abc.abstractmethod
    def __call__(
        self, x: Array, state: eqx.nn.State, *_, **__
    ) -> tuple[Array, eqx.nn.State]: ...
