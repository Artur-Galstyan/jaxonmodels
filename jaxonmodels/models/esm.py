import math

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Literal
from jaxonlayers.functions.activation import swiglu
from jaxtyping import Array, Float, Int, PRNGKeyArray

from jaxonmodels.functions import default_floating_dtype


class Affine3D:
    pass


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


def swiglu_ln_ffn(
    d_model: int,
    expansion_ratio: float,
    bias: bool,
    *,
    key: PRNGKeyArray,
    dtype: Any | None,
):
    key1, key2 = jax.random.split(key)
    return [
        eqx.nn.LayerNorm(d_model),
        eqx.nn.Linear(
            d_model,
            swiglu_correction_fn(expansion_ratio, d_model) * 2,
            use_bias=bias,
            key=key1,
            dtype=dtype,
        ),
        eqx.nn.Lambda(fn=swiglu),
        eqx.nn.Linear(
            swiglu_correction_fn(expansion_ratio, d_model),
            d_model,
            use_bias=bias,
            key=key2,
            dtype=dtype,
        ),
    ]


def gelu_ln_ffn(
    d_model: int,
    expansion_ratio: float,
    bias: bool,
    *,
    key: PRNGKeyArray,
    dtype: Any | None,
):
    hidden_dim = int(expansion_ratio * d_model)
    key1, key2 = jax.random.split(key)
    return [
        eqx.nn.LayerNorm(d_model),
        eqx.nn.Linear(d_model, hidden_dim, use_bias=bias, key=key1, dtype=dtype),
        eqx.nn.Lambda(fn=jax.nn.gelu),
        eqx.nn.Linear(hidden_dim, d_model, use_bias=bias, key=key2, dtype=dtype),
    ]


class GeometricReasoningOriginalImpl(eqx.Module):
    def __init__(
        self,
        c_s: int,
        v_heads: int,
        num_vector_messages: int = 1,
        mask_and_zero_frameless: bool = True,
        divide_residual_by_depth: bool = False,
        bias: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        pass


class ESMMultiHeadAttention(eqx.Module):
    d_model: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    qk_layernorm: bool = eqx.field(static=True)

    layernorm_qkv: list

    q_ln: eqx.nn.LayerNorm | None
    k_ln: eqx.nn.LayerNorm | None
    rotary: eqx.nn.RotaryPositionalEmbedding
    out_proj: eqx.nn.Linear

    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_bias: bool = False,
        qk_layernorm: bool = True,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qk_layernorm = qk_layernorm

        key, *subkeys = jax.random.split(key, 3)

        self.layernorm_qkv = [
            eqx.nn.LayerNorm(d_model),
            eqx.nn.Linear(
                d_model, d_model * 3, use_bias=use_bias, key=subkeys[0], dtype=dtype
            ),
        ]

        if qk_layernorm:
            self.q_ln = eqx.nn.LayerNorm(d_model, use_bias=use_bias)
            self.k_ln = eqx.nn.LayerNorm(d_model, use_bias=use_bias)
        else:
            self.q_ln = None
            self.k_ln = None

        self.rotary = eqx.nn.RotaryPositionalEmbedding(self.d_head)
        self.out_proj = eqx.nn.Linear(
            d_model, d_model, use_bias=use_bias, key=subkeys[1], dtype=dtype
        )

    def __call__(
        self,
        x: Float[Array, "seq_len d_model"],
        seq_id: Int[Array, "seq_len"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len d_model"]:
        seq_len = x.shape[0]
        dtype = x.dtype

        for l in self.layernorm_qkv:
            x = eqx.filter_vmap(l)(x)
        qkv = x
        q, k, v = jnp.split(qkv, 3, axis=-1)

        if self.q_ln is not None:
            q = eqx.filter_vmap(self.q_ln)(q).astype(dtype)
        if self.k_ln is not None:
            k = eqx.filter_vmap(self.k_ln)(k).astype(dtype)

        q = q.reshape(seq_len, self.n_heads, self.d_head)
        k = k.reshape(seq_len, self.n_heads, self.d_head)
        v = v.reshape(seq_len, self.n_heads, self.d_head)

        q = eqx.filter_vmap(self.rotary, in_axes=1, out_axes=1)(q)
        k = eqx.filter_vmap(self.rotary, in_axes=1, out_axes=1)(k)

        mask = None
        if seq_id is not None:
            mask = seq_id[:, None] == seq_id[None, :]
            mask = jnp.expand_dims(mask, axis=0)
        context = jax.nn.dot_product_attention(q, k, v, mask=mask)
        context = context.reshape(seq_len, self.d_model)

        output = eqx.filter_vmap(self.out_proj)(context)

        return output


def RegressionHead(
    d_model: int,
    output_dim: int,
    hidden_dim: int | None = None,
    *,
    key: PRNGKeyArray,
    dtype: Any | None = None,
) -> eqx.nn.Sequential:
    """Single-hidden layer MLP for supervised output.

    Args:
        d_model: input dimension
        output_dim: dimensionality of the output.
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
    Returns:
        output MLP module.
    """
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    key, subkey = jax.random.split(key)
    return eqx.nn.Sequential(
        [
            eqx.nn.Linear(d_model, hidden_dim, key=key),
            eqx.nn.Lambda(fn=jax.nn.gelu),
            eqx.nn.LayerNorm(hidden_dim),
            eqx.nn.Linear(hidden_dim, output_dim, key=subkey),
        ]
    )


class UnifiedTransformerBlock(eqx.Module):
    """
    A unified transformer block that can optionally incorporate geometric attention.

    This class defines a transformer block that can be configured to use
    geometric attention alongside the standard multi-head attention mechanism.
    It is designed to be a flexible component of transformer-based models, allowing for
    the integration of geometric reasoning.

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    n_layers : int
        The number of layers in the transformer block.
    use_geom_attn : bool, optional
        Whether to use geometric attention in addition to
        the standard multi-head attention. Defaults to False.
    v_heads : int, optional
        The number of heads to use for the geometric attention mechanism, if enabled.
        Must be specified if `use_geom_attn` is True.
    """

    attn: ESMMultiHeadAttention
    geom_attn: GeometricReasoningOriginalImpl
    ffn: list

    use_plain_attn: bool = eqx.field(static=True)
    use_geom_attn: bool = eqx.field(static=True)
    residue_scaling_factor: float = eqx.field(static=True)

    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_geom_attn: bool = False,
        use_plain_attn: bool = True,
        use_flash_attn: bool = False,
        v_heads: int | None = None,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        mask_and_zero_frameless: bool = False,
        qk_layernorm: bool = True,
        ffn_type: Literal["swiglu", "gelu"] = "swiglu",  # swiglu | gelu
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference
        self.use_plain_attn = use_plain_attn
        self.use_geom_attn = use_geom_attn
        key, mha_key = jax.random.split(key)
        if self.use_plain_attn:
            self.attn = ESMMultiHeadAttention(
                d_model,
                n_heads,
                bias,
                qk_layernorm=qk_layernorm,
                key=key,
                inference=inference,
                dtype=dtype,
            )

        if self.use_geom_attn:
            if v_heads is None:
                raise ValueError("v_heads must be specified when use_geom_attn is True")
            key, geom_key = jax.random.split(key)
            self.geom_attn = GeometricReasoningOriginalImpl(
                c_s=d_model,
                v_heads=v_heads,
                bias=bias,
                mask_and_zero_frameless=mask_and_zero_frameless,
                key=geom_key,
                dtype=dtype,
                inference=inference,
            )
        key, ffn_key = jax.random.split(key)
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(
                d_model,
                expansion_ratio,
                bias,
                key=key,
                dtype=dtype,
            )
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(
                d_model,
                expansion_ratio,
                bias,
                key=key,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.scaling_factor = residue_scaling_factor

    def __call__(
        self,
        x: Array,
        sequence_id: Array,
        frames: Affine3D,
        frames_mask: Array,
        chain_id: Array,
    ) -> Array:
        return x


class TransformerStack(eqx.Module):
    """
    A stack of transformer blocks used in the ESM-3 model.
    Each block is a UnifiedTransformerBlock,
    which can either be geometric attention or standard multi-head attention.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads.
        v_heads (int): The number of voting heads.
        n_layers (int): The number of transformer blocks in the stack.
        n_layers_geom (int, optional): The number of transformer blocks that use
            geometric attention.
        scale_residue (bool, optional): Whether to scale the residue connections
            in each transformer block.
        mask_and_zero_frameless (bool, optional): Whether to mask and zero frameless
            positions in the input.

        Only applies in the geometric attention blocks, which is
        conditioned on the structure
    """

    blocks: list[UnifiedTransformerBlock]
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int | None,
        n_layers: int,
        n_layers_geom: int = 1,
        scale_residue: bool = True,
        mask_and_zero_frameless: bool = False,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: Literal["swiglu", "gelu"] = "swiglu",  # swiglu | gelu
        expansion_ratio: float = 8 / 3,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        keys = jax.random.split(key, n_layers)
        self.blocks = [
            UnifiedTransformerBlock(
                d_model,
                n_heads,
                v_heads=v_heads,
                use_geom_attn=i < n_layers_geom,
                residue_scaling_factor=(
                    math.sqrt(n_layers / 36) if scale_residue else 1.0
                ),
                expansion_ratio=expansion_ratio,
                mask_and_zero_frameless=mask_and_zero_frameless,
                bias=bias,
                qk_layernorm=qk_layernorm,
                ffn_type=ffn_type,
                key=keys[i],
                dtype=dtype,
                inference=inference,
            )
            for i in range(n_layers)
        ]
        self.norm = eqx.nn.LayerNorm(d_model, use_bias=bias, dtype=dtype)


class ESMC(eqx.Module):
    """
    ESMC model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        n_layers (int): The number of transformer layers.
    """

    embed: eqx.nn.Embedding
    transformer: TransformerStack
    sequence_head: eqx.nn.Sequential

    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        # tokenizer: EsmSequenceTokenizer,
        # use_flash_attn: bool = True,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if not dtype:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference
        key, embed_key, transformer_stack_key, regression_head_key = jax.random.split(
            key, 4
        )
        self.embed = eqx.nn.Embedding(64, d_model, key=embed_key)

        self.transformer = TransformerStack(
            d_model,
            n_heads,
            None,
            n_layers,
            n_layers_geom=0,
            key=transformer_stack_key,
            dtype=dtype,
            inference=inference,
        )

        self.sequence_head = RegressionHead(d_model, 64, key=regression_head_key)
