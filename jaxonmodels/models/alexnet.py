import os
from pathlib import Path
from urllib.request import urlretrieve

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, Float, PRNGKeyArray
from statedict2pytree.converter import (
    autoconvert,
)

from jaxonmodels.functions import default_floating_dtype, dtype_to_str
from jaxonmodels.functions.utils import get_cache_path


class AlexNet(eqx.Module):
    features: list
    avgpool: eqx.nn.AdaptiveAvgPool2d
    classifier: list

    inference: bool

    def __init__(
        self,
        *,
        n_classes: int,
        key: PRNGKeyArray,
        inference: bool = False,
        dtype: Any | None = None,
    ):
        if not dtype:
            dtype = default_floating_dtype()
        assert dtype is not None

        _, *subkeys = jax.random.split(key, 10)
        self.features = [
            eqx.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=2,
                key=subkeys[0],
                dtype=dtype,
            ),
            eqx.nn.Lambda(fn=jax.nn.relu),
            eqx.nn.MaxPool2d(kernel_size=3, stride=2),
            eqx.nn.Conv2d(
                in_channels=64,
                out_channels=192,
                kernel_size=5,
                stride=1,
                padding=2,
                key=subkeys[1],
                dtype=dtype,
            ),
            eqx.nn.Lambda(fn=jax.nn.relu),
            eqx.nn.MaxPool2d(kernel_size=3, stride=2),
            eqx.nn.Conv2d(
                in_channels=192,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
                key=subkeys[2],
                dtype=dtype,
            ),
            eqx.nn.Lambda(fn=jax.nn.relu),
            eqx.nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                key=subkeys[3],
                dtype=dtype,
            ),
            eqx.nn.Lambda(fn=jax.nn.relu),
            eqx.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                key=subkeys[4],
                dtype=dtype,
            ),
            eqx.nn.Lambda(fn=jax.nn.relu),
            eqx.nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.classifier = [
            eqx.nn.Dropout(),
            eqx.nn.Linear(
                in_features=9216, out_features=4096, key=subkeys[5], dtype=dtype
            ),
            eqx.nn.Lambda(fn=jax.nn.relu),
            eqx.nn.Dropout(),
            eqx.nn.Linear(
                in_features=4096, out_features=4096, key=subkeys[6], dtype=dtype
            ),
            eqx.nn.Lambda(fn=jax.nn.relu),
            eqx.nn.Linear(
                in_features=4096, out_features=n_classes, key=subkeys[7], dtype=dtype
            ),
        ]

        self.inference = inference
        self.avgpool = eqx.nn.AdaptiveAvgPool2d((6, 6))

    def __call__(
        self,
        x: Float[Array, "c h w"],
        key: PRNGKeyArray | None = None,
    ) -> Array:
        if not self.inference:
            if not key:
                raise ValueError("Expected a random key in training mode, got None")

        for f in self.features:
            x = f(x)

        x = self.avgpool(x)
        x = jnp.ravel(x)

        for c in self.classifier:
            if key:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            x = c(x, key=subkey)

        return x

    @staticmethod
    def with_weights(
        model: str | None = None,
        key: PRNGKeyArray | None = None,
        dtype: Any | None = None,
    ) -> "AlexNet":
        """
        Loads alexnet. The model parameter is unused (there are no variants of AlexNet).
        If no key is provided, the default key (42) is used. Dtype defaults to float32.

        Returns:
            AlexNet
        """
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None

        if key is None:
            key = jax.random.key(42)

        dtype_str = dtype_to_str(dtype)
        alexnet = AlexNet(n_classes=1000, key=key, dtype=dtype)

        jaxonmodels_path = get_cache_path("alexnet")
        if os.path.exists(str(jaxonmodels_path / f"alexnet-{dtype_str}.eqx")):
            return eqx.tree_deserialise_leaves(
                str(jaxonmodels_path / f"alexnet-{dtype_str}.eqx"), alexnet
            )

        weights_url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
        weights_file = os.path.join(jaxonmodels_path, "alexnet-owt-7be5be79.pth")
        if not os.path.exists(weights_file):
            urlretrieve(weights_url, weights_file)

        import torch

        weights_dict = torch.load(
            weights_file, map_location=torch.device("cpu"), weights_only=True
        )

        alexnet = autoconvert(alexnet, weights_dict, dtype=dtype)

        eqx.tree_serialise_leaves(
            str(Path(jaxonmodels_path) / f"alexnet-{dtype_str}.eqx"), alexnet
        )

        return alexnet
