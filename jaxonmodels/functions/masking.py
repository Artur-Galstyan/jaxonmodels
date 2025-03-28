import jax.numpy as jnp


def canonical_mask(
    mask,
    mask_name,
    other_name="",
    other_type=None,
    target_type=jnp.float32,
    other_mask=None,
    check_other=True,
):
    if mask is None:
        return None
    if mask.dtype == bool:
        additive_mask = jnp.where(mask, -jnp.inf, 0.0).astype(target_type)
        return additive_mask
    elif jnp.issubdtype(mask.dtype, jnp.integer) or jnp.issubdtype(
        mask.dtype, jnp.floating
    ):
        return mask.astype(target_type)
    else:
        raise TypeError(
            f"{mask_name} must be bool, int, or float tensor, but got {mask.dtype}"
        )


def canonical_key_padding_mask(
    key_padding_mask, attn_mask=None, query_dtype=jnp.float32
):
    """Wrapper for canonicalizing key_padding_mask"""
    return canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_name="attn_mask",
        other_mask=attn_mask,
        target_type=query_dtype,
    )


def canonical_attn_mask(attn_mask, query_dtype=jnp.float32):
    """Wrapper for canonicalizing attn_mask"""
    return canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query_dtype,
        check_other=False,
    )


def build_attention_mask(context_length: int):
    mask = jnp.tril(jnp.zeros((context_length, context_length)))
    upper = jnp.triu(jnp.full((context_length, context_length), float("-inf")), k=1)

    mask = mask + upper
    return mask
