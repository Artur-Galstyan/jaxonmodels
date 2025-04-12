import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from equinox.nn import StatefulLayer
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions import patch_merging_pad, shifted_window_attention
from jaxonmodels.layers.regularization import StochasticDepth


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


class MLPWithDropout(eqx.nn.StatefulLayer):
    layers: list[eqx.Module]

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        key: PRNGKeyArray,
        norm_layer: Callable | None,
        activation: Callable | None,
        use_bias: bool,
        dropout: float,
        inference: bool,
        dtype: Any,
    ):
        assert dtype is not None

        key, *subkeys = jax.random.split(key, len(hidden_channels) + 1)
        layers = []
        in_dim = in_channels
        for i, hidden_dim in enumerate(hidden_channels[:-1]):
            layers.append(
                eqx.nn.Linear(
                    in_dim, hidden_dim, use_bias=use_bias, key=subkeys[i], dtype=dtype
                )
            )
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(
                eqx.nn.Lambda(fn=activation if activation is not None else lambda x: x)
            )
            layers.append(eqx.nn.Dropout(dropout, inference=inference))
            in_dim = hidden_dim

        layers.append(
            eqx.nn.Linear(
                in_dim,
                hidden_channels[-1],
                use_bias=use_bias,
                dtype=dtype,
                key=subkeys[-1],
            )
        )
        layers.append(eqx.nn.Dropout(dropout, inference=inference))
        self.layers = layers

    def __call__(
        self, x: Float[Array, "H W C"], state: eqx.nn.State
    ) -> tuple[Array, eqx.nn.State]:
        for layer in self.layers:
            if isinstance(layer, eqx.nn.StatefulLayer):
                x, state = layer(x, state)
            else:
                x = eqx.filter_vmap(eqx.filter_vmap(layer))(x)  # pyright: ignore

        return x, state


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
    logit_scale: Array | None
    relative_position_bias_table: Array | None
    relative_position_index: eqx.nn.StateIndex

    relative_coords_table: eqx.nn.StateIndex | None

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear

    cpb_mlp_ln1: eqx.nn.Linear | None
    cpb_mlp_ln2: eqx.nn.Linear | None
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

        self.cpb_mlp_ln1 = None
        self.cpb_mlp_ln2 = None
        self.logit_scale = None
        self.relative_coords_table = None

    def get_relative_position_bias(self, state: eqx.nn.State) -> Array:
        assert self.relative_position_bias_table is not None
        return get_relative_position_bias(
            self.relative_position_bias_table,
            state.get(self.relative_position_index),
            self.window_size,  # type: ignore[arg-type]
        )

    def __call__(
        self, x: Float[Array, "H W C"], state: eqx.nn.State
    ) -> tuple[Float[Array, "H W C"], eqx.nn.State]:
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
        ), state


class ShiftedWindowAttentionV2(ShiftedWindowAttention):
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
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
            key=key,
            inference=inference,
            dtype=dtype,
        )

        self.logit_scale = jnp.log(10 * jnp.ones((num_heads, 1, 1)))
        # mlp to generate continuous relative position bias
        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.cpb_mlp_ln1 = eqx.nn.Linear(
            2, 512, use_bias=True, key=subkey1, dtype=dtype
        )

        self.cpb_mlp_ln2 = eqx.nn.Linear(
            512, num_heads, use_bias=False, key=subkey2, dtype=dtype
        )
        if qkv_bias:
            assert self.qkv.bias is not None
            length = self.qkv.bias.size // 3
            new_bias = self.qkv.bias.at[length : 2 * length].set(
                jnp.zeros_like(self.qkv.bias[length : 2 * length])
            )
            where = lambda x: x.bias
            self.qkv = eqx.tree_at(where, self.qkv, new_bias)

        # get relative_coords_table
        relative_coords_h = jnp.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=dtype
        )
        relative_coords_w = jnp.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=dtype
        )
        relative_coords_table = jnp.stack(
            jnp.meshgrid(relative_coords_h, relative_coords_w, indexing="ij")
        )
        relative_coords_table = jnp.expand_dims(
            jnp.transpose(relative_coords_table, (1, 2, 0)), axis=0
        )  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table = relative_coords_table.at[:, :, :, 0].set(
            relative_coords_table[:, :, :, 0] / self.window_size[0] - 1
        )
        relative_coords_table = relative_coords_table.at[:, :, :, 1].set(
            relative_coords_table[:, :, :, 1] / self.window_size[1] - 1
        )

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            jnp.sign(relative_coords_table)
            * jnp.log2(jnp.abs(relative_coords_table) + 1.0)
            / 3.0
        )

        self.relative_coords_table = eqx.nn.StateIndex(relative_coords_table)
        self.relative_position_bias_table = None

    def get_relative_position_bias(self, state: eqx.nn.State) -> Array:
        assert self.relative_coords_table is not None
        assert self.cpb_mlp_ln1 is not None
        assert self.cpb_mlp_ln2 is not None
        relative_coords_table = state.get(self.relative_coords_table)
        relative_position_index = state.get(self.relative_position_index)

        cpb = eqx.filter_vmap(eqx.filter_vmap(eqx.filter_vmap(self.cpb_mlp_ln1)))(
            relative_coords_table
        )
        cpb = jax.nn.relu(cpb)
        cpb = eqx.filter_vmap(eqx.filter_vmap(eqx.filter_vmap(self.cpb_mlp_ln2)))(cpb)

        relative_position_bias = get_relative_position_bias(
            cpb.reshape(-1, self.num_heads),
            relative_position_index,
            self.window_size,
        )
        relative_position_bias = 16 * jax.nn.sigmoid(relative_position_bias)
        return relative_position_bias

    def __call__(
        self, x: Float[Array, "H W C"], state: eqx.nn.State
    ) -> tuple[Float[Array, "H W C"], eqx.nn.State]:
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
            logit_scale=self.logit_scale,
            inference=self.inference,
        ), state


class SwinTransformerBlock(eqx.Module):
    norm1: eqx.Module
    attn: eqx.Module

    stochastic_depth: StochasticDepth
    norm2: eqx.Module
    mlp: MLPWithDropout

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list[int],
        shift_size: list[int],
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., eqx.Module],
        attn_layer: Callable[..., eqx.Module],
        key: PRNGKeyArray,
        inference: bool,
        dtype: Any,
    ):
        self.norm1 = norm_layer(dim)
        key, subkey = jax.random.split(key)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
            qkv_bias=True,
            proj_bias=True,
            key=subkey,
            inference=inference,
            dtype=dtype,
        )
        self.stochastic_depth = StochasticDepth(
            stochastic_depth_prob, "row", inference=inference
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPWithDropout(
            dim,
            [int(dim * mlp_ratio), dim],
            norm_layer=None,
            use_bias=True,
            activation=jax.nn.gelu,
            dropout=dropout,
            inference=inference,
            key=key,
            dtype=dtype,
        )

        # todo: init properly

    def __call__(
        self, x: Float[Array, "H W C"], state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        key, subkey = jax.random.split(key)
        x_norm = eqx.filter_vmap(eqx.filter_vmap(self.norm1))(x)  # pyright: ignore
        attn, state = self.attn(x_norm, state)  # pyright: ignore
        x = x + self.stochastic_depth(attn, key=key)  # pyright: ignore
        x_norm = eqx.filter_vmap(eqx.filter_vmap(self.norm2))(x)  # pyright: ignore
        mlp_o, state = self.mlp(x_norm, state)  # pyright: ignore
        x = x + self.stochastic_depth(mlp_o, key=subkey)
        return x, state
