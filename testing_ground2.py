import abc

import equinox as eqx
import jax
from beartype.typing import Callable
from jaxtyping import Array


class CallableEqxModule(eqx.Module):
    @abc.abstractmethod
    def __call__(self, *_, **__):
        pass


class BarNorm(CallableEqxModule):
    x: Array

    def __call__(self, y):
        return self.x + y


class Foo(eqx.Module):
    bar: CallableEqxModule

    def __init__(self, bar: Callable[..., CallableEqxModule]):
        self.bar = bar(1)

    def __call__(self, x):
        return self.bar(x) + x


f = Foo(bar=BarNorm)
print(f(jax.numpy.array(5)))
