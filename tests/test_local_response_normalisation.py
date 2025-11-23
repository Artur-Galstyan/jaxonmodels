import equinox as eqx
import jax.numpy as jnp
import numpy as np
import torch
from jaxonlayers.layers import LocalResponseNormalization


def test_local_response_normalisation_against_pytorch():
    k = 2
    n = 5
    alpha = 1e-4
    beta = 0.75

    lrn_torch = torch.nn.LocalResponseNorm(size=n, alpha=alpha, beta=beta, k=k)
    lrn_jax = LocalResponseNormalization(k=k, n=n, alpha=alpha, beta=beta)

    b = 4
    c = 3
    h = 224
    w = 224
    i = np.random.uniform(size=(b, c, h, w))

    o_t = lrn_torch.forward(torch.from_numpy(i))
    o_j = eqx.filter_vmap(lrn_jax)(jnp.array(i))

    assert np.allclose(o_t.numpy(), np.array(o_j), atol=1e-4)
