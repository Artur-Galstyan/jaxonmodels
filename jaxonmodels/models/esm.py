import math

import equinox as eqx
import jax
from beartype.typing import Any, Literal
from jaxtyping import PRNGKeyArray

from jaxonmodels.functions import default_floating_dtype

d_model = 960
n_heads = 15
n_layers = 30


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
