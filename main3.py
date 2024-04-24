import numpy as np


torch_side = np.load("torch.resnet.conv1.npy")
jax_side = np.load("jax.resnet.conv1.npy")


print(np.allclose(torch_side, jax_side, atol=1e-4))
