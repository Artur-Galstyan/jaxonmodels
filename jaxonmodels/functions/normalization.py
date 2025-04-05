import jax.numpy as jnp


def normalize(x, p=2, axis=1, eps=1e-12):
    norm = jnp.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    norm = jnp.maximum(norm, eps)
    output = x / norm

    return output
