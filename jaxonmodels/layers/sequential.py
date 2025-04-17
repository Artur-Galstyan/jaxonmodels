import equinox as eqx
import jax
import jax.numpy as jnp


class BatchedLinear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, in_features, out_features, key, **kwargs):
        weight_key, bias_key = jax.random.split(key)

        self.weight = jax.random.normal(
            weight_key, (out_features, in_features)
        ) / jnp.sqrt(in_features)
        self.bias = jax.random.normal(bias_key, (out_features,)) * 0.1

        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x):
        input_shape = x.shape

        assert input_shape[-1] == self.weight.shape[1], (
            f"Expected last dimension to be {self.weight.shape[1]},"
            f" got {input_shape[-1]}"
        )

        if len(input_shape) > 1:
            batch_dims = input_shape[:-1]
            flattened_x = x.reshape(-1, self.in_features)
            result = flattened_x @ self.weight.T + self.bias
            return result.reshape(*batch_dims, self.out_features)
        else:
            return self.weight @ x + self.bias
