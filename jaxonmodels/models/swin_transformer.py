import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from equinox.nn import StatefulLayer
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions import patch_merging_pad, shifted_window_attention
from jaxonmodels.functions.utils import default_floating_dtype
from jaxonmodels.layers import LayerNorm, StochasticDepth


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
        self,
        x: Float[Array, "H W C"],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "H/2 W/2 C*2"], eqx.nn.State]:
        x = patch_merging_pad(x)
        if isinstance(self.norm, StatefulLayer):
            x, state = self.norm(x, state=state)  # pyright: ignore
        else:
            x = self.norm(x)  # pyright: ignore
        x = eqx.filter_vmap(eqx.filter_vmap(self.reduction))(x)  # ... H/2 W/2 2*C
        return x, state


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


class SwinTransformerBlock(StatefulLayer):
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


class SwinTransformerBlockV2(SwinTransformerBlock):
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
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
            key=key,
            inference=inference,
            dtype=dtype,
        )

    def __call__(
        self, x: Float[Array, "H W C"], state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        key, subkey = jax.random.split(key)

        # Apply attention first, then norm (different from V1)
        attn, state = self.attn(x, state)  # pyright: ignore
        attn_norm = eqx.filter_vmap(eqx.filter_vmap(self.norm1))(attn)  # pyright: ignore
        x = x + self.stochastic_depth(attn_norm, key=key)  # pyright: ignore

        # Apply MLP first, then norm (different from V1)
        mlp_o, state = self.mlp(x, state)  # pyright: ignore
        mlp_norm = eqx.filter_vmap(eqx.filter_vmap(self.norm2))(mlp_o)  # pyright: ignore
        x = x + self.stochastic_depth(mlp_norm, key=subkey)

        return x, state


class SwinTransformer(eqx.Module):
    features: eqx.nn.Sequential
    norm: eqx.Module
    avgpool: eqx.nn.AdaptiveAvgPool2d
    head: eqx.nn.Linear

    v: int = eqx.field(static=True)

    def __init__(
        self,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        num_heads: list[int],
        window_size: list[int],
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        stochastic_depth_prob: float,
        num_classes: int,
        norm_layer: Callable[..., eqx.Module] | None,
        block: Callable[..., eqx.Module] | None,
        downsample_layer: Callable[..., eqx.Module],
        attn_layer: Callable[..., eqx.Module],
        axis_name: str,
        inference: bool,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        if block is None:
            block = SwinTransformerBlock
        if block is SwinTransformerBlockV2:
            self.v = 2
        else:
            self.v = 1
        if norm_layer is None:
            norm_layer = functools.partial(LayerNorm, eps=1e-5, dtype=dtype)

        layers = []
        # split image into non-overlapping patches
        key, subkey = jax.random.split(key)
        fn = functools.partial(jnp.transpose, axes=[1, 2, 0])
        layers.append(
            eqx.nn.Sequential(
                [
                    eqx.nn.Conv2d(
                        3,
                        embed_dim,
                        kernel_size=(patch_size[0], patch_size[1]),
                        stride=(patch_size[0], patch_size[1]),
                        key=subkey,
                        dtype=dtype,
                    ),
                    eqx.nn.Lambda(fn=fn),
                    norm_layer(embed_dim),  # pyright: ignore
                ],
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                sd_prob = (
                    stochastic_depth_prob
                    * float(stage_block_id)
                    / (total_stage_blocks - 1)
                )
                key, subkey = jax.random.split(key)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[
                            0 if i_layer % 2 == 0 else w // 2 for w in window_size
                        ],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        attn_layer=attn_layer,
                        key=key,
                        inference=inference,
                        dtype=dtype,
                    )
                )
                stage_block_id += 1
            layers.append(eqx.nn.Sequential(stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                key, subkey = jax.random.split(key)
                layers.append(
                    downsample_layer(
                        dim,
                        functools.partial(norm_layer, 4 * dim),
                        dtype=dtype,
                        axis_name=axis_name,
                        inference=inference,
                        key=subkey,
                    )
                )
        self.features = eqx.nn.Sequential(layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        # self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = eqx.nn.AdaptiveAvgPool2d(1)
        key, subkey = jax.random.split(key)
        self.head = eqx.nn.Linear(num_features, num_classes, key=subkey, dtype=dtype)

        # todo: init properly

    def __call__(
        self, x: Array, state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        x, state = self.features(x, state, key=key)
        x = eqx.filter_vmap(eqx.filter_vmap(self.norm))(x)  # pyright: ignore
        x = jnp.transpose(x, (2, 0, 1))
        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.head(x)
        return x, state

    def model_order(self) -> list[str]:
        """
        Returns a list of JAX PyTree paths (as strings) for parameters and state,
        ordered to match the corresponding PyTorch state_dict keys.
        This is used by statedict2pytree for correct weight loading.
        """
        if self.v == 2:
            order = [
                "[0].features.layers[0].layers[0].weight",
                "[0].features.layers[0].layers[0].bias",
                "[0].features.layers[0].layers[2].weight",
                "[0].features.layers[0].layers[2].bias",
                "[0].features.layers[1].layers[0].norm1.weight",
                "[0].features.layers[1].layers[0].norm1.bias",
                "[0].features.layers[1].layers[0].attn.logit_scale",
                "[1][<flat index 1>]",
                "[1][<flat index 0>]",
                "[0].features.layers[1].layers[0].attn.qkv.weight",
                "[0].features.layers[1].layers[0].attn.qkv.bias",
                "[0].features.layers[1].layers[0].attn.proj.weight",
                "[0].features.layers[1].layers[0].attn.proj.bias",
                "[0].features.layers[1].layers[0].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[1].layers[0].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[1].layers[0].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[1].layers[0].norm2.weight",
                "[0].features.layers[1].layers[0].norm2.bias",
                "[0].features.layers[1].layers[0].mlp.layers[0].weight",
                "[0].features.layers[1].layers[0].mlp.layers[0].bias",
                "[0].features.layers[1].layers[0].mlp.layers[3].weight",
                "[0].features.layers[1].layers[0].mlp.layers[3].bias",
                "[0].features.layers[1].layers[1].norm1.weight",
                "[0].features.layers[1].layers[1].norm1.bias",
                "[0].features.layers[1].layers[1].attn.logit_scale",
                "[1][<flat index 3>]",
                "[1][<flat index 2>]",
                "[0].features.layers[1].layers[1].attn.qkv.weight",
                "[0].features.layers[1].layers[1].attn.qkv.bias",
                "[0].features.layers[1].layers[1].attn.proj.weight",
                "[0].features.layers[1].layers[1].attn.proj.bias",
                "[0].features.layers[1].layers[1].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[1].layers[1].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[1].layers[1].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[1].layers[1].norm2.weight",
                "[0].features.layers[1].layers[1].norm2.bias",
                "[0].features.layers[1].layers[1].mlp.layers[0].weight",
                "[0].features.layers[1].layers[1].mlp.layers[0].bias",
                "[0].features.layers[1].layers[1].mlp.layers[3].weight",
                "[0].features.layers[1].layers[1].mlp.layers[3].bias",
                "[0].features.layers[2].reduction.weight",
                "[0].features.layers[2].norm.weight",
                "[0].features.layers[2].norm.bias",
                "[0].features.layers[3].layers[0].norm1.weight",
                "[0].features.layers[3].layers[0].norm1.bias",
                "[0].features.layers[3].layers[0].attn.logit_scale",
                "[1][<flat index 5>]",
                "[1][<flat index 4>]",
                "[0].features.layers[3].layers[0].attn.qkv.weight",
                "[0].features.layers[3].layers[0].attn.qkv.bias",
                "[0].features.layers[3].layers[0].attn.proj.weight",
                "[0].features.layers[3].layers[0].attn.proj.bias",
                "[0].features.layers[3].layers[0].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[3].layers[0].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[3].layers[0].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[3].layers[0].norm2.weight",
                "[0].features.layers[3].layers[0].norm2.bias",
                "[0].features.layers[3].layers[0].mlp.layers[0].weight",
                "[0].features.layers[3].layers[0].mlp.layers[0].bias",
                "[0].features.layers[3].layers[0].mlp.layers[3].weight",
                "[0].features.layers[3].layers[0].mlp.layers[3].bias",
                "[0].features.layers[3].layers[1].norm1.weight",
                "[0].features.layers[3].layers[1].norm1.bias",
                "[0].features.layers[3].layers[1].attn.logit_scale",
                "[1][<flat index 7>]",
                "[1][<flat index 6>]",
                "[0].features.layers[3].layers[1].attn.qkv.weight",
                "[0].features.layers[3].layers[1].attn.qkv.bias",
                "[0].features.layers[3].layers[1].attn.proj.weight",
                "[0].features.layers[3].layers[1].attn.proj.bias",
                "[0].features.layers[3].layers[1].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[3].layers[1].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[3].layers[1].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[3].layers[1].norm2.weight",
                "[0].features.layers[3].layers[1].norm2.bias",
                "[0].features.layers[3].layers[1].mlp.layers[0].weight",
                "[0].features.layers[3].layers[1].mlp.layers[0].bias",
                "[0].features.layers[3].layers[1].mlp.layers[3].weight",
                "[0].features.layers[3].layers[1].mlp.layers[3].bias",
                "[0].features.layers[4].reduction.weight",
                "[0].features.layers[4].norm.weight",
                "[0].features.layers[4].norm.bias",
                "[0].features.layers[5].layers[0].norm1.weight",
                "[0].features.layers[5].layers[0].norm1.bias",
                "[0].features.layers[5].layers[0].attn.logit_scale",
                "[1][<flat index 9>]",
                "[1][<flat index 8>]",
                "[0].features.layers[5].layers[0].attn.qkv.weight",
                "[0].features.layers[5].layers[0].attn.qkv.bias",
                "[0].features.layers[5].layers[0].attn.proj.weight",
                "[0].features.layers[5].layers[0].attn.proj.bias",
                "[0].features.layers[5].layers[0].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[5].layers[0].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[5].layers[0].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[5].layers[0].norm2.weight",
                "[0].features.layers[5].layers[0].norm2.bias",
                "[0].features.layers[5].layers[0].mlp.layers[0].weight",
                "[0].features.layers[5].layers[0].mlp.layers[0].bias",
                "[0].features.layers[5].layers[0].mlp.layers[3].weight",
                "[0].features.layers[5].layers[0].mlp.layers[3].bias",
                "[0].features.layers[5].layers[1].norm1.weight",
                "[0].features.layers[5].layers[1].norm1.bias",
                "[0].features.layers[5].layers[1].attn.logit_scale",
                "[1][<flat index 11>]",
                "[1][<flat index 10>]",
                "[0].features.layers[5].layers[1].attn.qkv.weight",
                "[0].features.layers[5].layers[1].attn.qkv.bias",
                "[0].features.layers[5].layers[1].attn.proj.weight",
                "[0].features.layers[5].layers[1].attn.proj.bias",
                "[0].features.layers[5].layers[1].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[5].layers[1].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[5].layers[1].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[5].layers[1].norm2.weight",
                "[0].features.layers[5].layers[1].norm2.bias",
                "[0].features.layers[5].layers[1].mlp.layers[0].weight",
                "[0].features.layers[5].layers[1].mlp.layers[0].bias",
                "[0].features.layers[5].layers[1].mlp.layers[3].weight",
                "[0].features.layers[5].layers[1].mlp.layers[3].bias",
                "[0].features.layers[5].layers[2].norm1.weight",
                "[0].features.layers[5].layers[2].norm1.bias",
                "[0].features.layers[5].layers[2].attn.logit_scale",
                "[1][<flat index 13>]",
                "[1][<flat index 12>]",
                "[0].features.layers[5].layers[2].attn.qkv.weight",
                "[0].features.layers[5].layers[2].attn.qkv.bias",
                "[0].features.layers[5].layers[2].attn.proj.weight",
                "[0].features.layers[5].layers[2].attn.proj.bias",
                "[0].features.layers[5].layers[2].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[5].layers[2].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[5].layers[2].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[5].layers[2].norm2.weight",
                "[0].features.layers[5].layers[2].norm2.bias",
                "[0].features.layers[5].layers[2].mlp.layers[0].weight",
                "[0].features.layers[5].layers[2].mlp.layers[0].bias",
                "[0].features.layers[5].layers[2].mlp.layers[3].weight",
                "[0].features.layers[5].layers[2].mlp.layers[3].bias",
                "[0].features.layers[5].layers[3].norm1.weight",
                "[0].features.layers[5].layers[3].norm1.bias",
                "[0].features.layers[5].layers[3].attn.logit_scale",
                "[1][<flat index 15>]",
                "[1][<flat index 14>]",
                "[0].features.layers[5].layers[3].attn.qkv.weight",
                "[0].features.layers[5].layers[3].attn.qkv.bias",
                "[0].features.layers[5].layers[3].attn.proj.weight",
                "[0].features.layers[5].layers[3].attn.proj.bias",
                "[0].features.layers[5].layers[3].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[5].layers[3].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[5].layers[3].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[5].layers[3].norm2.weight",
                "[0].features.layers[5].layers[3].norm2.bias",
                "[0].features.layers[5].layers[3].mlp.layers[0].weight",
                "[0].features.layers[5].layers[3].mlp.layers[0].bias",
                "[0].features.layers[5].layers[3].mlp.layers[3].weight",
                "[0].features.layers[5].layers[3].mlp.layers[3].bias",
                "[0].features.layers[5].layers[4].norm1.weight",
                "[0].features.layers[5].layers[4].norm1.bias",
                "[0].features.layers[5].layers[4].attn.logit_scale",
                "[1][<flat index 17>]",
                "[1][<flat index 16>]",
                "[0].features.layers[5].layers[4].attn.qkv.weight",
                "[0].features.layers[5].layers[4].attn.qkv.bias",
                "[0].features.layers[5].layers[4].attn.proj.weight",
                "[0].features.layers[5].layers[4].attn.proj.bias",
                "[0].features.layers[5].layers[4].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[5].layers[4].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[5].layers[4].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[5].layers[4].norm2.weight",
                "[0].features.layers[5].layers[4].norm2.bias",
                "[0].features.layers[5].layers[4].mlp.layers[0].weight",
                "[0].features.layers[5].layers[4].mlp.layers[0].bias",
                "[0].features.layers[5].layers[4].mlp.layers[3].weight",
                "[0].features.layers[5].layers[4].mlp.layers[3].bias",
                "[0].features.layers[5].layers[5].norm1.weight",
                "[0].features.layers[5].layers[5].norm1.bias",
                "[0].features.layers[5].layers[5].attn.logit_scale",
                "[1][<flat index 19>]",
                "[1][<flat index 18>]",
                "[0].features.layers[5].layers[5].attn.qkv.weight",
                "[0].features.layers[5].layers[5].attn.qkv.bias",
                "[0].features.layers[5].layers[5].attn.proj.weight",
                "[0].features.layers[5].layers[5].attn.proj.bias",
                "[0].features.layers[5].layers[5].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[5].layers[5].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[5].layers[5].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[5].layers[5].norm2.weight",
                "[0].features.layers[5].layers[5].norm2.bias",
                "[0].features.layers[5].layers[5].mlp.layers[0].weight",
                "[0].features.layers[5].layers[5].mlp.layers[0].bias",
                "[0].features.layers[5].layers[5].mlp.layers[3].weight",
                "[0].features.layers[5].layers[5].mlp.layers[3].bias",
                "[0].features.layers[6].reduction.weight",
                "[0].features.layers[6].norm.weight",
                "[0].features.layers[6].norm.bias",
                "[0].features.layers[7].layers[0].norm1.weight",
                "[0].features.layers[7].layers[0].norm1.bias",
                "[0].features.layers[7].layers[0].attn.logit_scale",
                "[1][<flat index 21>]",
                "[1][<flat index 20>]",
                "[0].features.layers[7].layers[0].attn.qkv.weight",
                "[0].features.layers[7].layers[0].attn.qkv.bias",
                "[0].features.layers[7].layers[0].attn.proj.weight",
                "[0].features.layers[7].layers[0].attn.proj.bias",
                "[0].features.layers[7].layers[0].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[7].layers[0].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[7].layers[0].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[7].layers[0].norm2.weight",
                "[0].features.layers[7].layers[0].norm2.bias",
                "[0].features.layers[7].layers[0].mlp.layers[0].weight",
                "[0].features.layers[7].layers[0].mlp.layers[0].bias",
                "[0].features.layers[7].layers[0].mlp.layers[3].weight",
                "[0].features.layers[7].layers[0].mlp.layers[3].bias",
                "[0].features.layers[7].layers[1].norm1.weight",
                "[0].features.layers[7].layers[1].norm1.bias",
                "[0].features.layers[7].layers[1].attn.logit_scale",
                "[1][<flat index 23>]",
                "[1][<flat index 22>]",
                "[0].features.layers[7].layers[1].attn.qkv.weight",
                "[0].features.layers[7].layers[1].attn.qkv.bias",
                "[0].features.layers[7].layers[1].attn.proj.weight",
                "[0].features.layers[7].layers[1].attn.proj.bias",
                "[0].features.layers[7].layers[1].attn.cpb_mlp_ln1.weight",
                "[0].features.layers[7].layers[1].attn.cpb_mlp_ln1.bias",
                "[0].features.layers[7].layers[1].attn.cpb_mlp_ln2.weight",
                "[0].features.layers[7].layers[1].norm2.weight",
                "[0].features.layers[7].layers[1].norm2.bias",
                "[0].features.layers[7].layers[1].mlp.layers[0].weight",
                "[0].features.layers[7].layers[1].mlp.layers[0].bias",
                "[0].features.layers[7].layers[1].mlp.layers[3].weight",
                "[0].features.layers[7].layers[1].mlp.layers[3].bias",
                "[0].norm.weight",
                "[0].norm.bias",
                "[0].head.weight",
                "[0].head.bias",
            ]
        else:
            order = [
                "[0].features.layers[0].layers[0].weight",
                "[0].features.layers[0].layers[0].bias",
                "[0].features.layers[0].layers[2].weight",
                "[0].features.layers[0].layers[2].bias",
                "[0].features.layers[1].layers[0].norm1.weight",
                "[0].features.layers[1].layers[0].norm1.bias",
                "[0].features.layers[1].layers[0].attn.relative_position_bias_table",
                "[1][<flat index 0>]",
                "[0].features.layers[1].layers[0].attn.qkv.weight",
                "[0].features.layers[1].layers[0].attn.qkv.bias",
                "[0].features.layers[1].layers[0].attn.proj.weight",
                "[0].features.layers[1].layers[0].attn.proj.bias",
                "[0].features.layers[1].layers[0].norm2.weight",
                "[0].features.layers[1].layers[0].norm2.bias",
                "[0].features.layers[1].layers[0].mlp.layers[0].weight",
                "[0].features.layers[1].layers[0].mlp.layers[0].bias",
                "[0].features.layers[1].layers[0].mlp.layers[3].weight",
                "[0].features.layers[1].layers[0].mlp.layers[3].bias",
                # Stage 1, Block 1 (features.1.1 -> layers[1].layers[1])
                "[0].features.layers[1].layers[1].norm1.weight",
                "[0].features.layers[1].layers[1].norm1.bias",
                "[0].features.layers[1].layers[1].attn.relative_position_bias_table",
                "[1][<flat index 1>]",
                "[0].features.layers[1].layers[1].attn.qkv.weight",
                "[0].features.layers[1].layers[1].attn.qkv.bias",
                "[0].features.layers[1].layers[1].attn.proj.weight",
                "[0].features.layers[1].layers[1].attn.proj.bias",
                "[0].features.layers[1].layers[1].norm2.weight",
                "[0].features.layers[1].layers[1].norm2.bias",
                "[0].features.layers[1].layers[1].mlp.layers[0].weight",
                "[0].features.layers[1].layers[1].mlp.layers[0].bias",
                "[0].features.layers[1].layers[1].mlp.layers[3].weight",
                "[0].features.layers[1].layers[1].mlp.layers[3].bias",
                "[0].features.layers[2].reduction.weight",
                "[0].features.layers[2].norm.weight",
                "[0].features.layers[2].norm.bias",
                "[0].features.layers[3].layers[0].norm1.weight",
                "[0].features.layers[3].layers[0].norm1.bias",
                "[0].features.layers[3].layers[0].attn.relative_position_bias_table",
                "[1][<flat index 2>]",
                "[0].features.layers[3].layers[0].attn.qkv.weight",
                "[0].features.layers[3].layers[0].attn.qkv.bias",
                "[0].features.layers[3].layers[0].attn.proj.weight",
                "[0].features.layers[3].layers[0].attn.proj.bias",
                "[0].features.layers[3].layers[0].norm2.weight",
                "[0].features.layers[3].layers[0].norm2.bias",
                "[0].features.layers[3].layers[0].mlp.layers[0].weight",
                "[0].features.layers[3].layers[0].mlp.layers[0].bias",
                "[0].features.layers[3].layers[0].mlp.layers[3].weight",
                "[0].features.layers[3].layers[0].mlp.layers[3].bias",
                "[0].features.layers[3].layers[1].norm1.weight",
                "[0].features.layers[3].layers[1].norm1.bias",
                "[0].features.layers[3].layers[1].attn.relative_position_bias_table",
                "[1][<flat index 3>]",
                "[0].features.layers[3].layers[1].attn.qkv.weight",
                "[0].features.layers[3].layers[1].attn.qkv.bias",
                "[0].features.layers[3].layers[1].attn.proj.weight",
                "[0].features.layers[3].layers[1].attn.proj.bias",
                "[0].features.layers[3].layers[1].norm2.weight",
                "[0].features.layers[3].layers[1].norm2.bias",
                "[0].features.layers[3].layers[1].mlp.layers[0].weight",
                "[0].features.layers[3].layers[1].mlp.layers[0].bias",
                "[0].features.layers[3].layers[1].mlp.layers[3].weight",
                "[0].features.layers[3].layers[1].mlp.layers[3].bias",
                "[0].features.layers[4].reduction.weight",
                "[0].features.layers[4].norm.weight",
                "[0].features.layers[4].norm.bias",
                "[0].features.layers[5].layers[0].norm1.weight",
                "[0].features.layers[5].layers[0].norm1.bias",
                "[0].features.layers[5].layers[0].attn.relative_position_bias_table",
                "[1][<flat index 4>]",
                "[0].features.layers[5].layers[0].attn.qkv.weight",
                "[0].features.layers[5].layers[0].attn.qkv.bias",
                "[0].features.layers[5].layers[0].attn.proj.weight",
                "[0].features.layers[5].layers[0].attn.proj.bias",
                "[0].features.layers[5].layers[0].norm2.weight",
                "[0].features.layers[5].layers[0].norm2.bias",
                "[0].features.layers[5].layers[0].mlp.layers[0].weight",
                "[0].features.layers[5].layers[0].mlp.layers[0].bias",
                "[0].features.layers[5].layers[0].mlp.layers[3].weight",
                "[0].features.layers[5].layers[0].mlp.layers[3].bias",
                # Stage 3, Block 1 (features.5.1 -> layers[5].layers[1])
                "[0].features.layers[5].layers[1].norm1.weight",
                "[0].features.layers[5].layers[1].norm1.bias",
                "[0].features.layers[5].layers[1].attn.relative_position_bias_table",
                "[1][<flat index 5>]",
                "[0].features.layers[5].layers[1].attn.qkv.weight",
                "[0].features.layers[5].layers[1].attn.qkv.bias",
                "[0].features.layers[5].layers[1].attn.proj.weight",
                "[0].features.layers[5].layers[1].attn.proj.bias",
                "[0].features.layers[5].layers[1].norm2.weight",
                "[0].features.layers[5].layers[1].norm2.bias",
                "[0].features.layers[5].layers[1].mlp.layers[0].weight",
                "[0].features.layers[5].layers[1].mlp.layers[0].bias",
                "[0].features.layers[5].layers[1].mlp.layers[3].weight",
                "[0].features.layers[5].layers[1].mlp.layers[3].bias",
                # Stage 3, Block 2 (features.5.2 -> layers[5].layers[2])
                "[0].features.layers[5].layers[2].norm1.weight",
                "[0].features.layers[5].layers[2].norm1.bias",
                "[0].features.layers[5].layers[2].attn.relative_position_bias_table",
                "[1][<flat index 6>]",
                "[0].features.layers[5].layers[2].attn.qkv.weight",
                "[0].features.layers[5].layers[2].attn.qkv.bias",
                "[0].features.layers[5].layers[2].attn.proj.weight",
                "[0].features.layers[5].layers[2].attn.proj.bias",
                "[0].features.layers[5].layers[2].norm2.weight",
                "[0].features.layers[5].layers[2].norm2.bias",
                "[0].features.layers[5].layers[2].mlp.layers[0].weight",
                "[0].features.layers[5].layers[2].mlp.layers[0].bias",
                "[0].features.layers[5].layers[2].mlp.layers[3].weight",
                "[0].features.layers[5].layers[2].mlp.layers[3].bias",
                # Stage 3, Block 3 (features.5.3 -> layers[5].layers[3])
                "[0].features.layers[5].layers[3].norm1.weight",
                "[0].features.layers[5].layers[3].norm1.bias",
                "[0].features.layers[5].layers[3].attn.relative_position_bias_table",
                "[1][<flat index 7>]",
                "[0].features.layers[5].layers[3].attn.qkv.weight",
                "[0].features.layers[5].layers[3].attn.qkv.bias",
                "[0].features.layers[5].layers[3].attn.proj.weight",
                "[0].features.layers[5].layers[3].attn.proj.bias",
                "[0].features.layers[5].layers[3].norm2.weight",
                "[0].features.layers[5].layers[3].norm2.bias",
                "[0].features.layers[5].layers[3].mlp.layers[0].weight",
                "[0].features.layers[5].layers[3].mlp.layers[0].bias",
                "[0].features.layers[5].layers[3].mlp.layers[3].weight",
                "[0].features.layers[5].layers[3].mlp.layers[3].bias",
                # Stage 3, Block 4 (features.5.4 -> layers[5].layers[4])
                "[0].features.layers[5].layers[4].norm1.weight",
                "[0].features.layers[5].layers[4].norm1.bias",
                "[0].features.layers[5].layers[4].attn.relative_position_bias_table",
                "[1][<flat index 8>]",
                "[0].features.layers[5].layers[4].attn.qkv.weight",
                "[0].features.layers[5].layers[4].attn.qkv.bias",
                "[0].features.layers[5].layers[4].attn.proj.weight",
                "[0].features.layers[5].layers[4].attn.proj.bias",
                "[0].features.layers[5].layers[4].norm2.weight",
                "[0].features.layers[5].layers[4].norm2.bias",
                "[0].features.layers[5].layers[4].mlp.layers[0].weight",
                "[0].features.layers[5].layers[4].mlp.layers[0].bias",
                "[0].features.layers[5].layers[4].mlp.layers[3].weight",
                "[0].features.layers[5].layers[4].mlp.layers[3].bias",
                # Stage 3, Block 5 (features.5.5 -> layers[5].layers[5])
                "[0].features.layers[5].layers[5].norm1.weight",
                "[0].features.layers[5].layers[5].norm1.bias",
                "[0].features.layers[5].layers[5].attn.relative_position_bias_table",
                "[1][<flat index 9>]",
                "[0].features.layers[5].layers[5].attn.qkv.weight",
                "[0].features.layers[5].layers[5].attn.qkv.bias",
                "[0].features.layers[5].layers[5].attn.proj.weight",
                "[0].features.layers[5].layers[5].attn.proj.bias",
                "[0].features.layers[5].layers[5].norm2.weight",
                "[0].features.layers[5].layers[5].norm2.bias",
                "[0].features.layers[5].layers[5].mlp.layers[0].weight",
                "[0].features.layers[5].layers[5].mlp.layers[0].bias",
                "[0].features.layers[5].layers[5].mlp.layers[3].weight",
                "[0].features.layers[5].layers[5].mlp.layers[3].bias",
                # Downsample 3 (features.6 -> layers[6])
                "[0].features.layers[6].reduction.weight",
                "[0].features.layers[6].norm.weight",
                "[0].features.layers[6].norm.bias",
                # Stage 4, Block 0 (features.7.0 -> layers[7].layers[0])
                "[0].features.layers[7].layers[0].norm1.weight",
                "[0].features.layers[7].layers[0].norm1.bias",
                "[0].features.layers[7].layers[0].attn.relative_position_bias_table",
                "[1][<flat index 10>]",
                "[0].features.layers[7].layers[0].attn.qkv.weight",
                "[0].features.layers[7].layers[0].attn.qkv.bias",
                "[0].features.layers[7].layers[0].attn.proj.weight",
                "[0].features.layers[7].layers[0].attn.proj.bias",
                "[0].features.layers[7].layers[0].norm2.weight",
                "[0].features.layers[7].layers[0].norm2.bias",
                "[0].features.layers[7].layers[0].mlp.layers[0].weight",
                "[0].features.layers[7].layers[0].mlp.layers[0].bias",
                "[0].features.layers[7].layers[0].mlp.layers[3].weight",
                "[0].features.layers[7].layers[0].mlp.layers[3].bias",
                "[0].features.layers[7].layers[1].norm1.weight",
                "[0].features.layers[7].layers[1].norm1.bias",
                "[0].features.layers[7].layers[1].attn.relative_position_bias_table",
                "[1][<flat index 11>]",
                "[0].features.layers[7].layers[1].attn.qkv.weight",
                "[0].features.layers[7].layers[1].attn.qkv.bias",
                "[0].features.layers[7].layers[1].attn.proj.weight",
                "[0].features.layers[7].layers[1].attn.proj.bias",
                "[0].features.layers[7].layers[1].norm2.weight",
                "[0].features.layers[7].layers[1].norm2.bias",
                "[0].features.layers[7].layers[1].mlp.layers[0].weight",
                "[0].features.layers[7].layers[1].mlp.layers[0].bias",
                "[0].features.layers[7].layers[1].mlp.layers[3].weight",
                "[0].features.layers[7].layers[1].mlp.layers[3].bias",
                "[0].norm.weight",
                "[0].norm.bias",
                "[0].head.weight",
                "[0].head.bias",
            ]
        return order
