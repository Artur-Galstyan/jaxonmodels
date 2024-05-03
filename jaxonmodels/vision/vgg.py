import equinox as eqx
import jax
from beartype.typing import Any
from equinox.nn import Sequential
from jaxtyping import Array, PRNGKeyArray


cfgs: dict[str, list[str | int]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGGClassifier(eqx.Module):
    linear1: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    linear2: eqx.nn.Linear
    dropout2: eqx.nn.Dropout
    linear3: eqx.nn.Linear

    def __init__(
        self,
        dropout: float = 0.5,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key, *keys = jax.random.split(key, 4)
        self.linear1 = eqx.nn.Linear(512 * 7 * 7, 4096, key=keys[0])
        self.dropout1 = eqx.nn.Dropout(p=dropout)
        self.linear2 = eqx.nn.Linear(4096, 4096, key=keys[1])
        self.dropout2 = eqx.nn.Dropout(p=dropout)
        self.linear3 = eqx.nn.Linear(4096, 1000, key=keys[2])

    def __call__(self, x: Array, key1: PRNGKeyArray, key2: PRNGKeyArray) -> Array:
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.dropout1(x, key=key1)
        x = self.linear2(x)
        x = jax.nn.relu(x)
        x = self.dropout2(x, key=key2)
        x = self.linear3(x)
        return x


class VGG(eqx.Module):
    features: eqx.nn.Sequential
    avgpool: eqx.nn.AdaptiveAvgPool2d
    classifier: VGGClassifier

    def __init__(
        self,
        features: eqx.nn.Sequential,
        num_classes: int = 1000,
        dropout: float = 0.5,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.features = features
        self.avgpool = eqx.nn.AdaptiveAvgPool2d((7, 7))
        key, *keys = jax.random.split(key, 5)
        self.classifier = VGGClassifier(dropout=dropout, key=keys[3])

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        key1: PRNGKeyArray,
        key2: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        x, state = self.features(x, state, key=None)
        x = self.avgpool(x)
        x = x.reshape(-1)
        x = self.classifier(x, key1, key2)
        return x, state


def make_layers(
    cfg: list[str | int], batch_norm: bool = False, *, key: PRNGKeyArray
) -> Sequential:
    layers = []
    in_channels = 3
    relu = eqx.nn.Lambda(jax.nn.relu)
    for v in cfg:
        if v == "M":
            layers += [eqx.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = eqx.nn.Conv2d(in_channels, v, kernel_size=3, padding=1, key=key)

            if batch_norm:
                layers += [
                    conv2d,
                    eqx.nn.BatchNorm(v, axis_name="batch"),  # type: ignore
                    relu,
                ]
            else:
                layers += [conv2d, relu]
            in_channels = v
    return Sequential(layers)


def _vgg(
    cfg: str,
    batch_norm: bool,
    *,
    key: PRNGKeyArray,
    **kwargs: Any,
) -> tuple[VGG, eqx.nn.State]:
    layers_key, model_key = jax.random.split(key)
    model, state = eqx.nn.make_with_state(VGG)(
        make_layers(cfgs[cfg], batch_norm=batch_norm, key=layers_key),
        key=model_key,
        **kwargs,
    )

    return model, state


def vgg13(key: PRNGKeyArray, **kwargs: Any) -> tuple[VGG, eqx.nn.State]:
    return _vgg("B", False, key=key, **kwargs)
