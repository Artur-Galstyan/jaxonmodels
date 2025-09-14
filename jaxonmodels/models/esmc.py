import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from jaxonmodels.tokenizers.esm.tokenizers import EsmSequenceTokenizer


def RegressionHead(
    d_model: int,
    output_dim: int,
    key: PRNGKeyArray,
    hidden_dim: int | None = None,
) -> eqx.nn.Sequential:
    """Single-hidden layer MLP for supervised output.

    Args:
        d_model: input dimension
        output_dim: dimensionality of the output.
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
    Returns:
        output MLP module.
    """
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


class ESMC(eqx.Module):
    embed: eqx.nn.Embedding
    """
    ESMC model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        n_layers (int): The number of transformer layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        tokenizer: EsmSequenceTokenizer,
        *,
        key: PRNGKeyArray,
    ):
        self.embed = eqx.nn.Embedding(64, d_model)

        # self.transformer = TransformerStack(
        #     d_model,
        #     n_heads,
        #     None,
        #     n_layers,
        #     n_layers_geom=0,
        # )
        # key, subkey = jax.random.split(key)
        # self.sequence_head = RegressionHead(d_model, 64, subkey)
        # self.tokenizer = tokenizer
