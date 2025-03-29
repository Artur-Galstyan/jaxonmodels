import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray


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
