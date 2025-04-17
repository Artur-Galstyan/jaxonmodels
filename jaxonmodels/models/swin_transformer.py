import functools
import os
from pathlib import Path
from urllib.request import urlretrieve

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable, Literal
from equinox.nn import StatefulLayer
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from jaxonmodels.functions import patch_merging_pad, shifted_window_attention
from jaxonmodels.functions.utils import default_floating_dtype, dtype_to_str
from jaxonmodels.layers import LayerNorm, StochasticDepth
from jaxonmodels.layers.abstract import AbstractNorm, AbstractNormStateful
from jaxonmodels.statedict2pytree.model_orders import get_swin_model_order
from jaxonmodels.statedict2pytree.s2p import (
    convert,
    pytree_to_fields,
    serialize_pytree,
    state_dict_to_fields,
)

_SWIN_MODELS = {
    "swin_t_IMAGENET1K_V1": "https://download.pytorch.org/models/swin_t-704ceda3.pth",
    "swin_s_IMAGENET1K_V1": "https://download.pytorch.org/models/swin_s-5e29d889.pth",
    "swin_b_IMAGENET1K_V1": "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
    "swin_v2_t_IMAGENET1K_V1": "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
    "swin_v2_s_IMAGENET1K_V1": "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
    "swin_v2_b_IMAGENET1K_V1": "https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
}


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
    layers: list[PyTree]

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        key: PRNGKeyArray,
        norm_layer: Callable[..., AbstractNorm] | None,
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
                x = eqx.filter_vmap(eqx.filter_vmap(layer))(x)

        return x, state


class PatchMerging(StatefulLayer):
    reduction: eqx.nn.Linear
    norm: AbstractNorm | AbstractNormStateful

    inference: bool

    def __init__(
        self,
        dim: int,
        norm_layer: Callable[..., AbstractNorm | AbstractNormStateful],
        inference: bool,
        key: PRNGKeyArray,
        dtype: Any,
    ):
        self.inference = inference
        self.reduction = eqx.nn.Linear(
            4 * dim, 2 * dim, use_bias=False, key=key, dtype=dtype
        )
        self.norm = norm_layer(4 * dim)

    def __call__(
        self,
        x: Float[Array, "H W C"],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "H_half W_half C*2"], eqx.nn.State]:
        x = patch_merging_pad(x)
        if isinstance(self.norm, StatefulLayer):
            x, state = self.norm(x, state=state)
        else:
            x = self.norm(x)
        x = eqx.filter_vmap(eqx.filter_vmap(self.reduction))(x)  # ... H/2 W/2 2*C
        return x, state


class PatchMergingV2(StatefulLayer):
    reduction: eqx.nn.Linear
    norm: AbstractNorm | AbstractNormStateful

    inference: bool

    def __init__(
        self,
        dim: int,
        norm_layer: Callable[..., AbstractNorm | AbstractNormStateful],
        inference: bool,
        key: PRNGKeyArray,
        dtype: Any,
    ):
        self.inference = inference
        self.reduction = eqx.nn.Linear(
            4 * dim, 2 * dim, use_bias=False, key=key, dtype=dtype
        )
        self.norm = norm_layer(2 * dim)

    def __call__(
        self,
        x: Float[Array, "H W C"],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "H_half W_half C*2"], eqx.nn.State]:
        x = patch_merging_pad(x)
        x = eqx.filter_vmap(eqx.filter_vmap(self.reduction))(x)  # ... H/2 W/2 2*C
        if isinstance(self.norm, StatefulLayer):
            x, state = self.norm(x, state=state)
        else:
            x = self.norm(x)
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


# TODO: This is not the Eqx way of doing things!
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
                        norm_layer,
                        dtype=dtype,
                        inference=inference,
                        key=subkey,
                    )
                )
        self.features = eqx.nn.Sequential(layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
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
        return get_swin_model_order(self.v)


def _swin_transformer(
    key: PRNGKeyArray,
    inference: bool,
    num_classes: int,
    patch_size: list[int],
    embed_dim: int,
    depths: list[int],
    num_heads: list[int],
    window_size: list[int],
    stochastic_depth_prob: float,
    block: type[StatefulLayer],
    downsample_layer: type[PatchMerging | PatchMergingV2 | eqx.Module],
    attn_layer: type[ShiftedWindowAttention | ShiftedWindowAttentionV2 | eqx.Module],
    norm_layer: Callable | None = None,
    dtype: Any = None,
) -> tuple[SwinTransformer, eqx.nn.State]:
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None

    if norm_layer is None:
        norm_layer = functools.partial(LayerNorm, eps=1e-5, dtype=dtype)

    model, state = eqx.nn.make_with_state(SwinTransformer)(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        stochastic_depth_prob=stochastic_depth_prob,
        num_classes=num_classes,
        norm_layer=norm_layer,
        block=block,
        downsample_layer=downsample_layer,
        attn_layer=attn_layer,
        inference=inference,
        key=key,
        dtype=dtype,
    )
    return model, state


# Swin V1 Variants
def _swin_t(key, inference: bool, num_classes=1000, dtype: Any = None):
    return _swin_transformer(
        key=key,
        inference=inference,
        num_classes=num_classes,
        dtype=dtype,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlock,
        downsample_layer=PatchMerging,
        attn_layer=ShiftedWindowAttention,
    )


def _swin_s(key, inference: bool, num_classes=1000, dtype: Any = None):
    return _swin_transformer(
        key=key,
        inference=inference,
        num_classes=num_classes,
        dtype=dtype,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        block=SwinTransformerBlock,
        downsample_layer=PatchMerging,
        attn_layer=ShiftedWindowAttention,
    )


def _swin_b(key, inference: bool, num_classes=1000, dtype: Any = None):
    return _swin_transformer(
        key=key,
        inference=inference,
        num_classes=num_classes,
        dtype=dtype,
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        block=SwinTransformerBlock,
        downsample_layer=PatchMerging,
        attn_layer=ShiftedWindowAttention,
    )


def _swin_v2_t(key, inference: bool, num_classes=1000, dtype: Any = None):
    return _swin_transformer(
        key=key,
        inference=inference,
        num_classes=num_classes,
        dtype=dtype,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        attn_layer=ShiftedWindowAttentionV2,
    )


def _swin_v2_s(key, inference: bool, num_classes=1000, dtype: Any = None):
    return _swin_transformer(
        key=key,
        inference=inference,
        num_classes=num_classes,
        dtype=dtype,
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        attn_layer=ShiftedWindowAttentionV2,
    )


def _swin_v2_b(key, inference: bool, num_classes=1000, dtype: Any = None):
    return _swin_transformer(
        key=key,
        inference=inference,
        num_classes=num_classes,
        dtype=dtype,
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        attn_layer=ShiftedWindowAttentionV2,
    )


def _swin_with_weights(
    model: PyTree,
    weights: str,
    cache: bool,
    dtype: Any,
) -> tuple[SwinTransformer, eqx.nn.State]:
    dtype_str = dtype_to_str(dtype)

    weights_url = _SWIN_MODELS.get(weights)
    if weights_url is None:
        raise ValueError(f"No weights found for {weights}")

    jaxonmodels_dir = os.path.expanduser("~/.jaxonmodels/models")
    os.makedirs(jaxonmodels_dir, exist_ok=True)

    cache_filename = f"{weights}-{dtype_str}.eqx"
    cache_filepath = str(Path(jaxonmodels_dir) / cache_filename)

    if cache and os.path.exists(cache_filepath):
        print(f"Loading cached JAX model from: {cache_filepath}")
        return eqx.tree_deserialise_leaves(cache_filepath, model)  # pyright: ignore

    weights_dir = os.path.expanduser("~/.jaxonmodels/pytorch_weights")
    os.makedirs(weights_dir, exist_ok=True)
    filename = weights_url.split("/")[-1]
    weights_file = os.path.join(weights_dir, filename)
    if not os.path.exists(weights_file):
        print(f"Downloading weights from {weights_url} to {weights_file}")
        urlretrieve(weights_url, weights_file)
    else:
        print(f"Using existing PyTorch weights file: {weights_file}")

    print("Loading PyTorch state dict...")
    import torch

    weights_dict = torch.load(weights_file, map_location=torch.device("cpu"))
    if isinstance(weights_dict, dict) and "model" in weights_dict:
        weights_dict = weights_dict["model"]
    elif not isinstance(weights_dict, dict):
        raise TypeError(f"Expected state_dict but got {type(weights_dict)}")

    print("Converting PyTorch weights to JAX format...")

    torchfields = state_dict_to_fields(weights_dict)
    jaxfields, state_indices = pytree_to_fields(model)

    model = convert(
        weights_dict,
        model,
        jaxfields,
        state_indices,
        torchfields,
        dtype=dtype,
    )

    if cache:
        print(f"Caching JAX model to: {cache_filepath}")
        serialize_pytree(model, cache_filepath)

    return model


def _assert_model_and_weights_fit_swin(model_name: str, weights_name: str):
    if "_" in weights_name:
        weights_prefix = "_".join(weights_name.split("_")[:2])
        if weights_prefix.startswith("swin_v2"):
            weights_model = "_".join(weights_name.split("_")[:3])
        else:
            weights_model = weights_prefix

        if weights_model != model_name:
            raise ValueError(
                f"Model {model_name} is incompatible with weights {weights_name}. "
                f"Weight prefix '{weights_model}' does not match model name."
            )


def load_swin_transformer(
    model_name: Literal[
        "swin_t",
        "swin_s",
        "swin_b",  # V1
        "swin_v2_t",
        "swin_v2_s",
        "swin_v2_b",  # V2
    ],
    weights: Literal[
        "swin_t_IMAGENET1K_V1",
        "swin_s_IMAGENET1K_V1",
        "swin_b_IMAGENET1K_V1",
        "swin_v2_t_IMAGENET1K_V1",
        "swin_v2_s_IMAGENET1K_V1",
        "swin_v2_b_IMAGENET1K_V1",
    ]
    | None = None,
    num_classes: int = 1000,
    cache: bool = True,
    inference: bool = True,
    key: PRNGKeyArray | None = None,
    dtype: Any = None,
) -> tuple[SwinTransformer, eqx.nn.State]:
    if key is None:
        key = jax.random.key(42)

    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None

    model = None

    match model_name:
        case "swin_t":
            model = _swin_t(key, inference, num_classes, dtype=dtype)
        case "swin_s":
            model = _swin_s(key, inference, num_classes, dtype=dtype)
        case "swin_b":
            model = _swin_b(key, inference, num_classes, dtype=dtype)
        case "swin_v2_t":
            model = _swin_v2_t(key, inference, num_classes, dtype=dtype)
        case "swin_v2_s":
            model = _swin_v2_s(key, inference, num_classes, dtype=dtype)
        case "swin_v2_b":
            model = _swin_v2_b(key, inference, num_classes, dtype=dtype)
        case _:
            raise ValueError(f"Unknown Swin Transformer model name: {model_name}")

    if weights:
        _assert_model_and_weights_fit_swin(model_name, weights)
        model = _swin_with_weights(model, weights, cache, dtype=dtype)

    assert model is not None
    return model
