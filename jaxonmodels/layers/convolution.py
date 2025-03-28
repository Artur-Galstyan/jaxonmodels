import equinox as eqx
import jax
from beartype.typing import Callable, Sequence
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions.functions import (
    make_ntuple,
)
from jaxonmodels.layers.normalization import BatchNorm


class Conv2dNormActivation(eqx.Module):
    conv2d: eqx.nn.Conv2d
    norm: eqx.Module | None
    activation: eqx.nn.Lambda | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] | None = None,
        groups: int = 1,
        norm_layer: Callable[..., eqx.Module] | None = BatchNorm,
        activation_layer: Callable[..., Array] | None = jax.nn.relu,
        dilation: int | Sequence[int] = 1,
        use_bias: bool | None = None,
        dtype=None,
        axis_name: str = "batch",
        *,
        key: PRNGKeyArray,
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

        self.conv2d = eqx.nn.Conv2d(
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

        if norm_layer is not None:
            self.norm = norm_layer(out_channels, axis_name=axis_name)

        if activation_layer is not None:
            self.activation = eqx.nn.Lambda(activation_layer)
        else:
            self.activation = None

    def __call__(
        self,
        x: Float[Array, "c h w"],
        state: eqx.nn.State,
        inference: bool,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "c_out h_out w_out"], eqx.nn.State]:
        x = self.conv2d(x)

        if self.norm:
            x, state = self.norm(x, state, inference=inference)  # pyright: ignore

        if self.activation:
            x = self.activation(x)

        return x, state
