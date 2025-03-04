import os
from urllib.request import urlretrieve

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
from loguru import logger

from jaxonmodels.layers.local_response_normalisation import LocalResponseNormalization


class AlexNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    conv5: eqx.nn.Conv2d
    lrn1: LocalResponseNormalization
    lrn2: LocalResponseNormalization
    max_pool_1: eqx.nn.MaxPool2d
    max_pool_2: eqx.nn.MaxPool2d
    max_pool_3: eqx.nn.MaxPool2d

    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear

    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    final: eqx.nn.Linear

    def __init__(self, *, n_classes: int, key: jt.PRNGKeyArray):
        _, *subkeys = jax.random.split(key, 10)
        self.conv1 = eqx.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=11,
            stride=4,
            key=subkeys[0],
        )
        self.conv2 = eqx.nn.Conv2d(
            in_channels=64, out_channels=192, kernel_size=5, stride=1, key=subkeys[1]
        )
        self.conv3 = eqx.nn.Conv2d(
            in_channels=192, out_channels=384, kernel_size=3, stride=1, key=subkeys[2]
        )
        self.conv4 = eqx.nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, key=subkeys[3]
        )
        self.conv5 = eqx.nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, key=subkeys[4]
        )

        self.lrn1 = LocalResponseNormalization()
        self.lrn2 = LocalResponseNormalization()
        self.max_pool_1 = eqx.nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool_2 = eqx.nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool_3 = eqx.nn.MaxPool2d(kernel_size=3, stride=2)

        self.dense1 = eqx.nn.Linear(in_features=9216, out_features=4096, key=subkeys[5])
        self.dense2 = eqx.nn.Linear(in_features=4096, out_features=4096, key=subkeys[6])

        self.dropout1 = eqx.nn.Dropout()
        self.dropout2 = eqx.nn.Dropout()

        self.final = eqx.nn.Linear(
            in_features=4096, out_features=n_classes, key=subkeys[7]
        )

    def __call__(
        self,
        x: jt.Float[jt.Array, "c h w"],
        key: jt.PRNGKeyArray | None,
        inference: bool = False,
    ) -> jt.Array:
        if inference:
            key, subkey = None, None
        else:
            if not key:
                raise ValueError("Expected a random key in training mode, got None")
            else:
                key, subkey = jax.random.split(key)
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.lrn1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = self.lrn2(x)
        x = self.max_pool_2(x)

        x = self.conv3(x)
        x = jax.nn.relu(x)
        x = self.conv4(x)
        x = jax.nn.relu(x)
        x = self.conv5(x)
        x = jax.nn.relu(x)

        x = self.max_pool_3(x)
        x = jnp.ravel(x)

        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dropout1(x, key=key, inference=inference)
        x = self.dense2(x)
        x = jax.nn.relu(x)
        x = self.dropout2(x, key=subkey, inference=inference)

        x = self.final(x)
        return x

    @staticmethod
    def with_weights() -> "AlexNet":
        alexnet = AlexNet(n_classes=1000, key=jax.random.key(0))

        # from pytorch
        weights_url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"

        weights_dir = os.path.expanduser("~/.jaxonmodels/weights")
        os.makedirs(weights_dir, exist_ok=True)

        weights_file = os.path.join(weights_dir, "alexnet-owt-7be5be79.pth")
        if not os.path.exists(weights_file):
            logger.info(f"Downloading AlexNet weights to {weights_file}...")
            urlretrieve(weights_url, weights_file)
            logger.info("Download complete.")
        else:
            logger.info(f"AlexNet weights already exist at {weights_file}.")

        import torch

        weights_dict = torch.load(
            weights_file, map_location=torch.device("cpu"), weights_only=True
        )

        print(weights_dict.keys())
        for k in weights_dict.keys():
            print(k, weights_dict[k].shape)

        return alexnet
