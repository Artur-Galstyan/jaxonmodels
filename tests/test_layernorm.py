import jax.numpy as jnp
import numpy as np
import torch
from jaxonlayers.layers import LayerNorm


def test_layernorm():
    np.random.seed(42)
    x = np.array(np.random.normal(size=(56, 56, 96)))
    torch_layer_norm = torch.nn.LayerNorm(96)
    t_out = torch_layer_norm(torch.Tensor(x)).detach().numpy()
    jax_layer_norm2 = LayerNorm(96)
    j_out2 = jax_layer_norm2(jnp.array(x))

    print(np.allclose(t_out, np.array(j_out2), atol=1e-5))
