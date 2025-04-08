import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from equinox.nn import StatefulLayer
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions import patch_merging_pad
from jaxonmodels.functions.attention import shifted_window_attention


def get_relative_position_bias(
    relative_position_bias_table: Array,
    relative_position_index: Array,
    window_size: list[int],
) -> Array:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.reshape(N, N, -1)
    relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))
    return relative_position_bias


class PatchMerging(StatefulLayer):
    reduction: eqx.nn.Linear
    norm: eqx.Module

    inference: bool

    def __init__(
        self,
        dim: int,
        norm_layer: Callable[..., eqx.Module],
        inference: bool,
        axis_name: str | None,
        key: PRNGKeyArray,
        dtype: Any,
    ):
        self.inference = inference
        self.reduction = eqx.nn.Linear(
            4 * dim, 2 * dim, use_bias=False, key=key, dtype=dtype
        )
        self.norm = norm_layer()

    def __call__(
        self, x: Float[Array, "H W C"], state: eqx.nn.State
    ) -> tuple[Float[Array, "H/2 W/2 C*2"], eqx.nn.State]:
        x = patch_merging_pad(x)
        if self.is_stateful():
            x, state = self.norm(x, state=state)  # pyright: ignore
        else:
            x = self.norm(x)  # pyright: ignore
        x = eqx.filter_vmap(eqx.filter_vmap(self.reduction))(x)  # ... H/2 W/2 2*C
        return x, state

    def is_stateful(self) -> bool:
        return True if isinstance(self.norm, StatefulLayer) else False


class ShiftedWindowAttention(eqx.nn.StatefulLayer):
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear

    relative_position_bias_table: Array
    relative_position_index: eqx.nn.StateIndex

    inference: bool

    window_size: list[int] = eqx.field(static=True)
    shift_size: list[int] = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    attention_dropout: float = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        window_size: list[int],
        shift_size: list[int],
        num_heads: int,
        qkv_bias: bool,
        proj_bias: bool,
        attention_dropout: float,
        dropout: float,
        key: PRNGKeyArray,
        inference: bool,
        dtype: Any,
    ):
        self.inference = inference
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.qkv = eqx.nn.Linear(
            dim, dim * 3, use_bias=qkv_bias, key=subkey1, dtype=dtype
        )
        self.proj = eqx.nn.Linear(
            dim, dim, use_bias=proj_bias, key=subkey2, dtype=dtype
        )

        self.relative_position_bias_table = jnp.zeros(
            shape=(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ),
            dtype=dtype,
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = jnp.arange(self.window_size[0])
        coords_w = jnp.arange(self.window_size[1])
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = jnp.reshape(coords, (coords.shape[0], -1))  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = jnp.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        relative_coords = relative_coords.at[:, :, 0].set(
            relative_coords[:, :, 0] + self.window_size[0] - 1
        )  # shift to start from 0
        relative_coords = relative_coords.at[:, :, 1].set(
            relative_coords[:, :, 1] + self.window_size[1] - 1
        )
        relative_coords = relative_coords.at[:, :, 0].set(
            relative_coords[:, :, 0] * 2 * self.window_size[1] - 1
        )
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.relative_position_index = eqx.nn.StateIndex(relative_position_index)

    def get_relative_position_bias(self, state: eqx.nn.State) -> Array:
        return get_relative_position_bias(
            self.relative_position_bias_table,
            state.get(self.relative_position_index),
            self.window_size,  # type: ignore[arg-type]
        )

    def __call__(self, x: Array, state: eqx.nn.State) -> Array:
        relative_position_bias = self.get_relative_position_bias(state)
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            inference=self.inference,
        )
