import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from esm.layers.attention import MultiHeadAttention
from statedict2pytree.converter import autoconvert

from jaxonmodels.models.esm import ESMMultiHeadAttention


@pytest.mark.parametrize(
    "d_model, n_heads, seq_len, bias, qk_layernorm",
    [
        (96, 4, 7, True, True),
        (960, 15, 7, False, True),
    ],
)
def test_mha_implementations(
    d_model: int, n_heads: int, seq_len, bias: bool, qk_layernorm: bool
):
    np.random.seed(42)
    x = np.random.uniform(size=(1, seq_len, d_model))
    x = np.array(x, dtype=np.float32)
    seq_id = np.array([[True for _ in range(seq_len)]])

    torch_mha = MultiHeadAttention(d_model, n_heads, bias, qk_layernorm=qk_layernorm)
    state_dict = torch_mha.state_dict()
    torch_output = torch_mha(torch.from_numpy(x), torch.from_numpy(seq_id))

    key = jax.random.key(42)
    jax_mha = ESMMultiHeadAttention(d_model, n_heads, bias, key=key)
    jax_mha = autoconvert(jax_mha, state_dict)
    x = jnp.array(x, dtype=jnp.float32)
    seq_id = jnp.array([[True for _ in range(seq_len)]])

    jax_output = eqx.filter_vmap(jax_mha)(x, seq_id)

    torch_output = torch_output.detach().numpy()
    jax_output = np.array(jax_output)

    assert np.allclose(torch_output, jax_output, atol=1e-5)
