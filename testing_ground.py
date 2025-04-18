import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxonmodels.layers.sequential import BatchedLinear
from jaxonmodels.statedict2pytree.s2p import autoconvert

x = np.ones(shape=(4, 4, 8))

torch_linear = torch.nn.Linear(8, 16)
eqx_linear = eqx.nn.Linear(8, 16, key=jax.random.key(22))
eqx_linear = autoconvert(eqx_linear, torch_linear.state_dict())
eqx_batched_linear = BatchedLinear(8, 16, key=jax.random.key(22))
eqx_batched_linear = autoconvert(eqx_batched_linear, torch_linear.state_dict())

t_out = torch_linear(torch.Tensor(x))
e_out1 = eqx.filter_vmap(eqx.filter_vmap(eqx_linear))(jnp.array(x))
e_out2 = eqx_batched_linear(jnp.array(x))

print("torch vs eqx filtervmap", np.allclose(t_out.detach().numpy(), np.array(e_out1)))
print("torch vs batched linear", np.allclose(t_out.detach().numpy(), np.array(e_out2)))
