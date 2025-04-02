import equinox as eqx
import jax
from beartype.typing import Any, Callable, Sequence
from equinox.nn import BatchNorm, StatefulLayer
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions import (
    make_ntuple,
)


class ConvNormActivation(StatefulLayer):
    conv: eqx.nn.Conv
    norm: eqx.Module | None
    activation: eqx.nn.Lambda | None

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] | None = None,
        groups: int = 1,
        activation_layer: Callable[..., Array] | None = jax.nn.relu,
        dilation: int | Sequence[int] = 1,
        use_bias: bool | None = None,
        axis_name: str = "batch",
        *,
        norm_layer: Callable[..., eqx.Module] | None,
        key: PRNGKeyArray,
        dtype: Any,
    ):
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = (
                    len(kernel_size)
                    if isinstance(kernel_size, Sequence)
                    else len(dilation)  # pyright: ignore
                )
                kernel_size = make_ntuple(kernel_size, _conv_dim)
                dilation = make_ntuple(dilation, _conv_dim)
                padding = tuple(
                    (kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim)
                )
        if use_bias is None:
            use_bias = norm_layer is None

        key, subkey = jax.random.split(key)

        self.conv = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            dtype=dtype,
            key=subkey,
        )

        self.norm = None
        if norm_layer is not None:
            if norm_layer == BatchNorm:
                self.norm = norm_layer(out_channels, axis_name=axis_name, dtype=dtype)
            else:
                self.norm = norm_layer(out_channels, dtype=dtype)

        if activation_layer is not None:
            self.activation = eqx.nn.Lambda(activation_layer)
        else:
            self.activation = None

    def __call__(
        self,
        x: Float[Array, "c *num_spatial_dims"],
        state: eqx.nn.State | None,
        inference: bool | None,
        key: PRNGKeyArray | None = None,
    ) -> (
        Float[Array, "c_out *num_spatial_dims_out"]
        | tuple[Float[Array, "c_out *num_spatial_dims_out"], eqx.nn.State]
    ):
        x = self.conv(x)

        if self.norm:
            x, state = self.norm(x, state, inference=inference)  # pyright: ignore

        if self.activation:
            x = self.activation(x)

        return x, state
