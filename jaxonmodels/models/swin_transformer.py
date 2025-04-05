import equinox as eqx
from beartype.typing import Any, Callable
from equinox.nn import StatefulLayer
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions import patch_merging_pad


class PatchMerging(StatefulLayer):
    reduction: eqx.nn.Linear
    norm: eqx.Module

    inference: bool

    def __init__(
        self,
        dim: int,
        norm_layer: Callable[..., eqx.Module],
        inference: bool,
        axis_name: str | None,
        key: PRNGKeyArray,
        dtype: Any,
    ):
        self.inference = inference
        self.reduction = eqx.nn.Linear(
            4 * dim, 2 * dim, use_bias=False, key=key, dtype=dtype
        )
        self.norm = norm_layer()

    def __call__(
        self, x: Float[Array, "H W C"], state: eqx.nn.State
    ) -> tuple[Float[Array, "H/2 W/2 C*2"], eqx.nn.State]:
        x = patch_merging_pad(x)
        if self.is_stateful():
            x, state = self.norm(x, state=state)  # pyright: ignore
        else:
            x = self.norm(x)  # pyright: ignore
        x = eqx.filter_vmap(eqx.filter_vmap(self.reduction))(x)  # ... H/2 W/2 2*C
        return x, state

    def is_stateful(self) -> bool:
        return True if isinstance(self.norm, StatefulLayer) else False
