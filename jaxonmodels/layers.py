import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Callable, Hashable, Sequence
from equinox.nn import State
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions.functions import (
    canonical_mask,
    make_ntuple,
    multi_head_attention_forward,
    selective_scan,
)


class BatchNorm(eqx.nn.StatefulLayer):
    state_index: eqx.nn.StateIndex

    gamma: Float[Array, "size"] | None
    beta: Float[Array, "size"] | None

    inference: bool
    axis_name: Hashable | Sequence[Hashable]

    size: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    affine: bool = eqx.field(static=True)

    def __init__(
        self,
        size: int,
        axis_name: Hashable | Sequence[Hashable],
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        inference: bool = False,
    ):
        self.size = size
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.inference = inference
        self.axis_name = axis_name

        self.gamma = jnp.ones(self.size) if self.affine else None
        self.beta = jnp.zeros(self.size) if self.affine else None

        self.state_index = eqx.nn.StateIndex((jnp.zeros(size), jnp.ones(size)))

    def __call__(
        self,
        x: Array,
        state: State,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Array, State]:
        if inference is None:
            inference = self.inference

        running_mean, running_var = state.get(self.state_index)

        input_shape = x.shape
        ndim = len(input_shape)

        if ndim == 1:
            batch_mean = jax.lax.pmean(x, axis_name=self.axis_name)
            batch_size = jax.lax.psum(1, axis_name=self.axis_name)

            if inference:
                x_normalized = (x - running_mean) / jnp.sqrt(running_var + self.eps)
            else:
                xmu = x - batch_mean
                sq = xmu**2
                batch_var = jax.lax.pmean(sq, axis_name=self.axis_name)
                std = jnp.sqrt(batch_var + self.eps)
                x_normalized = xmu / std

                correction_factor = batch_size / jnp.maximum(batch_size - 1, 1)
                running_mean = (
                    1 - self.momentum
                ) * running_mean + self.momentum * batch_mean
                running_var = (1 - self.momentum) * running_var + self.momentum * (
                    batch_var * correction_factor
                )

                state = state.set(self.state_index, (running_mean, running_var))
        else:
            spatial_axes = tuple(range(1, ndim))  # All dims except channel dim (0)

            if inference:
                x_normalized = (
                    x - running_mean.reshape((-1,) + (1,) * (ndim - 1))
                ) / jnp.sqrt(running_var.reshape((-1,) + (1,) * (ndim - 1)) + self.eps)
            else:
                spatial_mean = jnp.mean(x, axis=spatial_axes)

                batch_mean = jax.lax.pmean(spatial_mean, axis_name=self.axis_name)
                batch_size = jax.lax.psum(1, axis_name=self.axis_name)

                broadcast_shape = (-1,) + (1,) * (ndim - 1)
                batch_mean_broadcasted = batch_mean.reshape(broadcast_shape)

                xmu = x - batch_mean_broadcasted
                sq = xmu**2

                spatial_var = jnp.mean(sq, axis=spatial_axes)
                batch_var = jax.lax.pmean(spatial_var, axis_name=self.axis_name)

                batch_var_broadcasted = batch_var.reshape(broadcast_shape)
                std = jnp.sqrt(batch_var_broadcasted + self.eps)

                x_normalized = xmu / std

                spatial_size = 1
                for dim in spatial_axes:
                    spatial_size *= x.shape[dim]
                total_size = batch_size * spatial_size

                correction_factor = total_size / jnp.maximum(total_size - 1, 1)
                running_mean = (
                    1 - self.momentum
                ) * running_mean + self.momentum * batch_mean
                running_var = (1 - self.momentum) * running_var + self.momentum * (
                    batch_var * correction_factor
                )

                state = state.set(self.state_index, (running_mean, running_var))

        out = x_normalized
        if self.affine and self.gamma is not None and self.beta is not None:
            if ndim > 1:
                broadcast_shape = (-1,) + (1,) * (ndim - 1)
                gamma_broadcasted = self.gamma.reshape(broadcast_shape)
                beta_broadcasted = self.beta.reshape(broadcast_shape)
                out = gamma_broadcasted * x_normalized + beta_broadcasted
            else:
                out = self.gamma * x_normalized + self.beta

        return out, state


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


class LocalResponseNormalization(eqx.Module):
    k: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __init__(self, k=2, n=5, alpha=1e-4, beta=0.75) -> None:
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        c, _, _ = x.shape
        p = jnp.pad(x, pad_width=[(self.n // 2, self.n // 2), (0, 0), (0, 0)])

        def _body(i):
            window = jax.lax.dynamic_slice_in_dim(p, i, self.n) ** 2
            d = (jnp.einsum("ijk->jk", window) * self.alpha + self.k) ** self.beta
            b = x[i] / d
            return b

        ys = eqx.filter_vmap(_body)(jnp.arange(c))
        return ys


class MultiheadAttention(eqx.Module):
    q_proj_weight: Array | None
    k_proj_weight: Array | None
    v_proj_weight: Array | None

    in_proj_weight: Array | None

    in_proj_bias: Array | None

    out_proj: eqx.nn.Linear

    bias_k: Array | None
    bias_v: Array | None

    embed_dim: int = eqx.field(static=True)
    kdim: int = eqx.field(static=True)
    vdim: int = eqx.field(static=True)
    _qkv_same_embed_dim: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    add_zero_attn: bool = eqx.field(static=True)

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        uniform_initializer = jax.nn.initializers.uniform()

        if not self._qkv_same_embed_dim:
            key, *subkeys = jax.random.split(key, 4)
            self.q_proj_weight = uniform_initializer(
                key=subkeys[0], shape=(embed_dim, embed_dim)
            )
            self.k_proj_weight = uniform_initializer(
                key=subkeys[1], shape=(embed_dim, self.kdim)
            )
            self.v_proj_weight = uniform_initializer(
                key=subkeys[2], shape=(embed_dim, self.vdim)
            )
            self.in_proj_weight = None
        else:
            key, subkey = jax.random.split(key)
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None
            self.in_proj_weight = uniform_initializer(
                key=subkey, shape=(3 * embed_dim, embed_dim)
            )

        if bias:
            self.in_proj_bias = jnp.empty((3 * embed_dim))
        else:
            self.in_proj_bias = None
        key, subkey = jax.random.split(key)
        out_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=bias, key=subkey)
        if bias:
            assert out_proj.bias is not None
            new_bias = jnp.zeros_like(out_proj.bias)
            where = lambda l: l.bias
            self.out_proj = eqx.tree_at(where, out_proj, new_bias)
        else:
            self.out_proj = out_proj

        if add_bias_kv:
            normal_initializer = jax.nn.initializers.normal()
            key, *subkeys = jax.random.split(key, 3)
            self.bias_k = normal_initializer(key=subkeys[0], shape=(1, 1, embed_dim))
            self.bias_v = normal_initializer(key=subkeys[0], shape=(1, 1, embed_dim))
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        key_padding_mask: Array | None = None,
        need_weights: bool = True,
        attn_mask: Array | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        inference: bool = False,
    ) -> tuple[Array, Array | None]:
        key_padding_mask = canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                inference=inference,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                inference=inference,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        return attn_output, attn_output_weights


class SelectiveStateSpaceModel(eqx.Module):
    input_proj: eqx.nn.Linear
    delta_proj: eqx.nn.Linear
    A_log: Float[Array, "d_inner d_state"]
    D: Float[Array, " d_inner"]

    d_inner: int = eqx.field(static=True)
    dt_rank: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)

    def __init__(
        self,
        d_inner: int,
        dt_rank: int,
        d_state: int,
        use_input_proj_bias: bool = False,
        use_delta_proj_bias: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        (
            key,
            input_proj_key,
            delta_proj_key,
        ) = jax.random.split(key, 3)
        self.input_proj = eqx.nn.Linear(
            d_inner,
            dt_rank + d_state * 2,
            use_bias=use_input_proj_bias,
            key=input_proj_key,
        )

        self.delta_proj = eqx.nn.Linear(
            dt_rank, d_inner, use_bias=use_delta_proj_bias, key=delta_proj_key
        )
        A = jnp.repeat(jnp.arange(1, d_state + 1), d_inner).reshape(d_inner, d_state)
        self.A_log = jnp.log(A)
        self.D = jnp.ones(d_inner)

    def __call__(self, x: Float[Array, "seq_length d_inner"]):
        A = -jnp.exp(self.A_log)
        D = self.D

        delta_b_c = jax.vmap(self.input_proj)(x)

        split_indices = [
            self.dt_rank,
            self.dt_rank + self.d_state,
        ]
        delta, B, C = jnp.split(delta_b_c, split_indices, axis=-1)
        delta = jax.nn.softplus(jax.vmap(self.delta_proj)(delta))

        y = selective_scan(x, delta, A, B, C, D)
        return y
