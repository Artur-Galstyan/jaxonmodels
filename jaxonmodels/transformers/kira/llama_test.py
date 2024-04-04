from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree import leaves, map
from jaxtyping import Array, PyTree
from kira.model_args import LLaMAModelArgs
from kira.models.llama.llama2 import LLaMA
from memory_profiler import profile


def find_min_max(pytree: PyTree) -> tuple[float, float]:
    return min(map(lambda x: jnp.min(x), leaves(pytree))), max(
        map(lambda x: jnp.max(x), leaves(pytree))
    )


def quantize(pytree: PyTree, bits: Literal[8, 16]) -> PyTree:
    quantized_dtype = None

    match bits:
        case 8:
            quantized_dtype = jnp.uint8
        case 16:
            quantized_dtype = jnp.float16

    pytree = eqx.filter(pytree, eqx.is_inexact_array)
    min_val, max_val = find_min_max(pytree)
    scale = (max_val - min_val) / (2**bits - 1)

    def quantize_array(x: Array) -> Array:
        return jnp.array(jnp.round(x / scale) * scale, dtype=quantized_dtype)

    return map(quantize_array, pytree)


@profile
def main():
    llama_args = LLaMAModelArgs(dim=4096, n_layers=4, n_heads=4, vocab_size=384)

    llama, state = eqx.nn.make_with_state(LLaMA)(
        model_args=llama_args, key=jax.random.PRNGKey(-1)
    )

    # llama = quantize(llama, 8)

    llama = quantize(llama, 16)

    # test_x = jnp.ones((8,), dtype=jnp.int32)
    # y, state = llama(test_x, state, key=None, inference=True)
    # print(y.shape)


if __name__ == "__main__":
    main()
