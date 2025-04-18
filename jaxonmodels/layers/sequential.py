import math

import equinox as eqx
import jax
from beartype.typing import Literal
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions.utils import default_floating_dtype, default_init


class BatchedLinear(eqx.Module):
    weight: Array
    bias: Array | None
    in_features: int | Literal["scalar"] = eqx.field(static=True)
    out_features: int | Literal["scalar"] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int | Literal["scalar"],
        out_features: int | Literal["scalar"],
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        weight_key, bias_key = jax.random.split(key)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        if in_features_ == 0:
            lim = 1.0
        else:
            lim = 1 / math.sqrt(in_features_)
        wshape = (out_features_, in_features_)
        self.weight = default_init(weight_key, wshape, dtype, lim)
        bshape = (out_features_,)
        self.bias = default_init(bias_key, bshape, dtype, lim) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(
        self, x: Float[Array, "*batch in_features"]
    ) -> Float[Array, "*batch out_features"]:
        input_shape = x.shape

        assert input_shape[-1] == self.weight.shape[1], (
            f"Expected last dimension to be {self.weight.shape[1]},"
            f" got {input_shape[-1]}"
        )

        if len(input_shape) > 1:
            batch_dims = input_shape[:-1]
            flattened_x = x.reshape(-1, self.in_features)
            result = flattened_x @ self.weight.T
            if self.use_bias and self.bias is not None:
                result = result + self.bias
            return result.reshape(*batch_dims, self.out_features)
        else:
            result = self.weight @ x
            if self.use_bias and self.bias is not None:
                result = result + self.bias
            return result
