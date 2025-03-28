import jax
from jaxtyping import Array, PRNGKeyArray


def stochastic_depth(
    input: Array,
    p: float,
    mode: str,
    inference: bool,
    key: PRNGKeyArray,
) -> Array:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if inference or p == 0.0:
        return input
    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = jax.random.bernoulli(key, p=survival_rate, shape=size).astype(input.dtype)
    if survival_rate > 0.0:
        noise = noise / survival_rate
    return input * noise
