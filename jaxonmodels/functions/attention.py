import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from jaxonmodels.functions.normalization import normalize


def multi_head_attention_forward(
    query: Float[Array, "tgt_len d_model"],
    key: Float[Array, "src_len d_model"],
    value: Float[Array, "src_len d_model"],
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Float[Array, "3*d_model d_model"] | None = None,
    in_proj_bias: Float[Array, "3*d_model"] | None = None,
    bias_k: Float[Array, "1 d_model"] | None = None,
    bias_v: Float[Array, "1 d_model"] | None = None,
    add_zero_attn: bool = False,
    dropout_p: float = 0.0,
    out_proj_weight: Float[Array, "d_model d_model"] | None = None,
    out_proj_bias: Float[Array, "d_model"] | None = None,
    inference: bool = False,
    key_padding_mask: Float[Array, "src_len"] | Bool[Array, "src_len"] | None = None,
    attn_mask: Float[Array, "tgt_len src_len"] | None = None,
    need_weights: bool = True,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Float[Array, "d_model d_model"] | None = None,
    k_proj_weight: Float[Array, "d_model d_model"] | None = None,
    v_proj_weight: Float[Array, "d_model d_model"] | None = None,
    static_k: Float[Array, "src_len d_model"] | None = None,
    static_v: Float[Array, "src_len d_model"] | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    dropout_key: PRNGKeyArray | None = None,
) -> tuple[
    Float[Array, "tgt_len d_model"],
    Float[Array, "num_heads tgt_len src_len"]
    | Float[Array, "tgt_len src_len"]
    | Float[Array, "tgt_len src_len+1"]
    | None,
]:
    tgt_len, d_model = query.shape
    src_len, k_dim = key.shape
    value_len, v_dim = value.shape

    assert d_model == k_dim == v_dim == embed_dim_to_check, (
        "Embedding dimensions must match"
    )

    assert src_len == value_len, "Key and value must have the same sequence length"

    head_dim = d_model // num_heads
    assert head_dim * num_heads == d_model, "embed_dim must be divisible by num_heads"

    if dropout_p > 0.0:
        assert dropout_key is not None, (
            "dropout_key must be provided if dropout_p > 0.0"
        )

    if use_separate_proj_weight:
        # When using separate projection weights for q, k, v
        assert q_proj_weight is not None, (
            "q_proj_weight should not be None when use_separate_proj_weight=True"
        )
        assert k_proj_weight is not None, (
            "k_proj_weight should not be None when use_separate_proj_weight=True"
        )
        assert v_proj_weight is not None, (
            "v_proj_weight should not be None when use_separate_proj_weight=True"
        )

        q = query @ q_proj_weight.T

        if static_k is None:
            k = key @ k_proj_weight.T
        else:
            k = static_k
            src_len, _ = k.shape

        if static_v is None:
            v = value @ v_proj_weight.T
        else:
            v = static_v
            value_len, _ = v.shape

        if in_proj_bias is not None:
            q_bias, k_bias, v_bias = jnp.split(in_proj_bias, 3)
            q = q + q_bias
            k = k + k_bias
            v = v + v_bias

    else:
        assert in_proj_weight is not None, (
            "in_proj_weight should not be None when use_separate_proj_weight=False"
        )

        q_proj_weight_part, k_proj_weight_part, v_proj_weight_part = jnp.split(
            in_proj_weight, 3
        )

        q = query @ q_proj_weight_part.T

        if static_k is None:
            k = key @ k_proj_weight_part.T
        else:
            k = static_k
            src_len, _ = static_k.shape

        if static_v is None:
            v = value @ v_proj_weight_part.T
        else:
            v = static_v
            value_len, _ = static_v.shape

        if in_proj_bias is not None:
            q_bias, k_bias, v_bias = jnp.split(in_proj_bias, 3)
            q = q + q_bias
            k = k + k_bias
            v = v + v_bias

    assert src_len == value_len

    q = q.reshape(tgt_len, num_heads, head_dim)
    k = k.reshape(src_len, num_heads, head_dim)
    v = v.reshape(src_len, num_heads, head_dim)

    if add_zero_attn:
        zero_attn_shape = (1, num_heads, head_dim)
        k_zeros = jnp.zeros(zero_attn_shape)
        v_zeros = jnp.zeros(zero_attn_shape)

        k = jnp.concatenate([k, k_zeros], axis=0)
        v = jnp.concatenate([v, v_zeros], axis=0)

        src_len += 1
        value_len += 1

    if bias_k is not None and bias_v is not None:
        bias_k = bias_k.reshape(1, num_heads, head_dim)
        bias_v = bias_v.reshape(1, num_heads, head_dim)

        k = jnp.concatenate([k, bias_k], axis=0)
        v = jnp.concatenate([v, bias_v], axis=0)

        src_len += 1
        value_len += 1

    assert src_len == value_len

    # [tgt_len, num_heads, head_dim] → [num_heads, tgt_len, head_dim]
    q = jnp.transpose(q, (1, 0, 2))

    # [src_len, num_heads, head_dim] → [num_heads, src_len, head_dim]
    k = jnp.transpose(k, (1, 0, 2))
    v = jnp.transpose(v, (1, 0, 2))

    scale = jnp.sqrt(head_dim)
    attn_output_weights = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / scale

    if key_padding_mask is not None:
        padding_mask = key_padding_mask.reshape(1, 1, src_len)
        padding_mask = jnp.repeat(padding_mask, num_heads, axis=0)
        padding_mask = jnp.repeat(padding_mask, tgt_len, axis=1)
        attn_output_weights = jnp.where(
            padding_mask, float("-inf"), attn_output_weights
        )

    if attn_mask is not None:
        # [tgt_len, src_len] -> [num_heads, tgt_len, src_len]
        mask = attn_mask.reshape(1, tgt_len, src_len)
        mask = jnp.repeat(mask, num_heads, axis=0)
        attn_output_weights = attn_output_weights + mask

    if is_causal:
        causal_mask = jnp.triu(jnp.ones((tgt_len, src_len)), k=1)
        causal_mask = (causal_mask == 1).reshape(1, tgt_len, src_len)
        causal_mask = jnp.repeat(causal_mask, num_heads, axis=0)
        attn_output_weights = jnp.where(causal_mask, float("-inf"), attn_output_weights)

    # [num_heads, tgt_len, src_len]
    attn_output_weights = jax.nn.softmax(attn_output_weights, axis=-1)

    if dropout_p > 0.0 and not inference:
        assert dropout_key is not None, (
            "dropout_key required because dropout_p > 0.0 and training"
        )
        dropout_mask = jax.random.bernoulli(
            dropout_key, 1 - dropout_p, attn_output_weights.shape
        )
        scale = 1.0 / (1.0 - dropout_p)
        attn_output_weights = attn_output_weights * dropout_mask * scale

    attn_output = jnp.matmul(attn_output_weights, v)
    attn_output = jnp.transpose(attn_output, (1, 0, 2))
    attn_output = attn_output.reshape(tgt_len, d_model)

    assert out_proj_weight is not None, "out_proj_weight must be provided"
    attn_output = attn_output @ out_proj_weight.T

    if out_proj_bias is not None:
        attn_output = attn_output + out_proj_bias

    if need_weights:
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(axis=0)
        return attn_output, attn_output_weights
    else:
        return attn_output, None


def shifted_window_attention(
    x: Float[Array, "H W C"],
    qkv_weight: Float[Array, "in_dim out_dim"],
    proj_weight: Float[Array, "out_dim out_dim"],
    relative_position_bias: Array,
    window_size: list[int],
    num_heads: int,
    shift_size: list[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Array | None = None,
    proj_bias: Array | None = None,
    logit_scale: Array | None = None,
    inference: bool = False,
) -> Float[Array, "H W C"]:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        x (Array[H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Array[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Array[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Array): The learned relative position
        bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Array[out_dim], optional): The bias tensor of query, key, value.
        proj_bias (Array[out_dim], optional): The bias tensor of projection.
        logit_scale (Array[out_dim], optional): Logit scale of cosine attention for
        Swin Transformer V2
        inference (bool, optional): Training flag used by the dropout parameters.
    Returns:
        Array[H, W, C]: The output tensor after shifted window attention.
    """
    H, W, C = x.shape
    to_pad_W = (window_size[1] - W % window_size[1]) % window_size[1]
    to_pad_H = (window_size[0] - H % window_size[0]) % window_size[0]
    x = jnp.pad(x, ((0, to_pad_H), (0, to_pad_W), (0, 0)))
    pad_H, pad_W, _ = x.shape
    print(f"{to_pad_W=}, {to_pad_H=}, {x.shape=}, {pad_H=}, {pad_W=}")

    shift_size = shift_size.copy()
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    print(window_size, shift_size)

    # cyclic shift
    if sum(shift_size) > 0:
        x = jnp.roll(x, shift=(-shift_size[0], -shift_size[1]), axis=(0, 1))

    print(f"{x.shape=}")

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    print(f"{num_windows=}")
    x = jnp.reshape(
        x,
        (
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
            C,
        ),
    )
    print(f"{x.shape=}")
    x = jnp.transpose(x, (0, 2, 1, 3, 4)).reshape(
        num_windows, window_size[0] * window_size[1], C
    )
    print(f"{x.shape=}")

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        length = qkv_bias.size // 3
        qkv_bias = qkv_bias.at[length : 2 * length].set(0.0)
        print(f"{qkv_bias.shape=}")

    def linear(x: Array, weight: Array, bias: Array | None):
        if bias is None:
            return weight @ x
        else:
            return (weight @ x) + bias

    linear_pt = functools.partial(linear, weight=qkv_weight, bias=qkv_bias)

    qkv = eqx.filter_vmap(eqx.filter_vmap(linear_pt))(x)
    print(f"{qkv.shape=}")
    win_size, patches, _ = qkv.shape
    qkv = jnp.transpose(
        qkv.reshape(win_size, patches, 3, num_heads, C // num_heads), (2, 0, 3, 1, 4)
    )
    print(f"{qkv.shape=}")
    q, k, v = qkv[0], qkv[1], qkv[2]
    print(f"{q.shape=}, {k.shape=}, {v.shape=}")
    if logit_scale is not None:
        # cosine attention
        print(f"{normalize(q, axis=-1).shape=}")
        print(f"{jnp.transpose(normalize(k, axis=-1), (0, 1, 3, 2)).shape=}")
        attn = normalize(q, axis=-1) @ jnp.transpose(
            normalize(k, axis=-1), (0, 1, 3, 2)
        )
        print(f"{attn.shape=}")
        logit_scale = jnp.exp(jnp.minimum(logit_scale, jnp.log(100.0)))
        print(f"{logit_scale=}")
        attn = attn * logit_scale
        print(f"{attn.shape=}")
    else:
        q = q * (C // num_heads) ** -0.5
        print(f"{q.shape=}")
        attn = q @ (jnp.transpose(normalize(k, axis=-1), (0, 1, 3, 2)))
        print(f"{attn.shape=}")
    # add relative position bias
    attn = attn + relative_position_bias
    print(f"{attn.shape=}")

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = jnp.zeros((pad_H, pad_W), dtype=x.dtype)
        print(f"{attn_mask.shape=}")
        h_slices = (
            (0, -window_size[0]),
            (-window_size[0], -shift_size[0]),
            (-shift_size[0], None),
        )
        w_slices = (
            (0, -window_size[1]),
            (-window_size[1], -shift_size[1]),
            (-shift_size[1], None),
        )
        print(f"{h_slices=}, {w_slices=}")
    #     count = 0
    #     for h in h_slices:
    #         for w in w_slices:
    #             attn_mask[h[0] : h[1], w[0] : w[1]] = count
    #             count += 1
    #     attn_mask = attn_mask.view(
    #         pad_H // window_size[0],
    #         window_size[0],
    #         pad_W // window_size[1],
    #         window_size[1],
    #     )
    #     attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
    #         num_windows, window_size[0] * window_size[1]
    #     )
    #     attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
    #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
    #         attn_mask == 0, float(0.0)
    #     )
    #     attn = attn.view(
    #         x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1)
    #     )
    #     attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
    #     attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    # attn = F.softmax(attn, dim=-1)
    # attn = F.dropout(attn, p=attention_dropout, training=training)

    # x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    # x = F.linear(x, proj_weight, proj_bias)
    # x = F.dropout(x, p=dropout, training=training)

    # # reverse windows
    # x = x.view(
    #     B,
    #     pad_H // window_size[0],
    #     pad_W // window_size[1],
    #     window_size[0],
    #     window_size[1],
    #     C,
    # )
    # x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # # reverse cyclic shift
    # if sum(shift_size) > 0:
    #     x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # # unpad features
    # x = x[:, :H, :W, :].contiguous()
    return x
