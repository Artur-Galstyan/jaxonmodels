from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree


def generate_text(
    model: PyTree,
    max_seq_len: int,
    max_new_tokens: int,
    decode: Callable,
    vocab_size: int,
    print_to_console: bool = True,
    random_key_seed: int = 0,
    state: Optional[eqx.nn.State] = None,
) -> tuple[list[str], list[int]]:
    jitted_model = eqx.filter_jit(model)
    x = jnp.zeros((max_seq_len,), dtype=jnp.int32)
    key = jax.random.PRNGKey(random_key_seed)
    tokens = []
    decoded_tokens = []
    for _ in range(max_new_tokens):
        key, subkey, model_key = jax.random.split(key, 3)
        if state is None:
            logits = jitted_model(x, state=state, key=model_key)
        else:
            logits, state = jitted_model(x, state=state, key=model_key)
        logits = logits[-1, :]
        probs = jax.nn.softmax(logits, axis=-1)

        next_token = jax.random.choice(
            subkey,
            jnp.arange(len(probs)),
            p=probs,
        )
        next_token = jnp.array(next_token, dtype=jnp.int32).reshape((1,))
        next_token = min(next_token.item(), vocab_size - 1)

        if print_to_console:
            print(decode(np.array([next_token])), end="")

        tokens.append(next_token)
        decoded_tokens.append(decode(np.array([next_token])))

        x = jnp.concatenate((x[1:], jnp.array([next_token])))
    return decoded_tokens, tokens
