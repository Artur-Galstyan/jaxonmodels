import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


class DropoutClassifier(eqx.Module):
    dropout: eqx.nn.Dropout
    linear: eqx.nn.Linear

    def __init__(
        self, p: float, in_features: int, out_features: int, key: PRNGKeyArray
    ):
        self.dropout = eqx.nn.Dropout(p=p)
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)

    def __call__(self, x: Array, inference: bool, key: PRNGKeyArray) -> Array:
        x = self.dropout(x, inference=inference, key=key)
        x = self.linear(x)

        return x
