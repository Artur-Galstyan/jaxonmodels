import re

import jax
import numpy as np
import torch
from jaxtyping import PyTree
from pydantic import BaseModel


class ChunkifiedPytreePath(BaseModel):
    path: str


class ChunkifiedStatedictPath(BaseModel):
    path: str


class Field(BaseModel):
    path: str
    shape: tuple[int, ...]
    skip: bool = False


class TorchField(Field):
    pass


class JaxField(Field):
    type: str


def can_reshape(shape1: tuple, shape2: tuple):
    """
    Check if two shapes can be reshaped to each other.

    Args:
        shape1 (tuple): First shape.
        shape2 (tuple): Second shape.

    Returns:
        bool: True if shapes can be reshaped to each other, False otherwise.
    """
    product1 = np.prod(shape1)
    product2 = np.prod(shape2)

    return product1 == product2


def get_node(tree: PyTree, targets: list[str]) -> PyTree | None:
    """
    Retrieve a node from the PyTree based on the given path.

    Args:
        tree (PyTree): The PyTree to search.
        targets (list[str]): Path to the target node.

    Returns:
        PyTree | None: The target node if found, None otherwise.

    Examples:
    ```python
    import equinox as eqx

    class MyModel(eqx.Module):
        layer1: eqx.nn.Linear
        layer2: eqx.nn.Linear

    model = MyModel(
            layer1=eqx.nn.Linear(10, 20, key=jax.random.key(0)),
            layer2=eqx.nn.Linear(20, 5, key=jax.random.key(1)),
        )
    layer1_weight = get_node(model, ['layer1', 'weight'])
    assert layer1_weight.shape == (20, 10)
    nonexistent_node = get_node(model, ['layer3'])
    assert nonexistent_node is None
    ```
    """
    if len(targets) == 0 or tree is None:
        return tree
    else:
        next_target: str = targets[0]
        if bool(re.search(r"\[\d+\]", next_target)):
            split_index = next_target.rfind("[")
            name, index = next_target[:split_index], next_target[split_index:]
            index = index[1:-1]
            if hasattr(tree, name):
                subtree = getattr(tree, name)[int(index)]
            else:
                subtree = None
        else:
            if hasattr(tree, next_target):
                subtree = getattr(tree, next_target)
            else:
                subtree = None
        return get_node(subtree, targets[1:])


def state_dict_to_fields(state_dict: dict[str, torch.Tensor]) -> list[TorchField]:
    """
    Convert a PyTorch state dict to a list of TorchField objects.

    Args:
        state_dict (Optional[dict]): The PyTorch state dict to be converted.

    Returns:
        list[TorchField]: A list of TorchField objects representing the state dict.
    """
    if state_dict is None:
        return []
    fields: list[TorchField] = []
    for key, value in state_dict.items():
        if hasattr(value, "shape") and len(value.shape) > 0:
            fields.append(TorchField(path=key, shape=tuple(value.shape)))
    return fields


def pytree_to_fields(pytree: PyTree) -> list[JaxField]:
    """
    Convert a JAX PyTree to a list of JaxField objects.

    Args:
        pytree (PyTree): The JAX PyTree to be converted.

    Returns:
        list[JaxField]: A list of JaxField objects representing the PyTree.
    """
    flattened, _ = jax.tree_util.tree_flatten_with_path(pytree)
    fields = []
    for key_path, value in flattened:
        path = jax.tree_util.keystr(key_path)
        type_path = path.split(".")[1:-1]
        target_path = path.split(".")[1:]
        node_type = type(get_node(pytree, type_path))
        node = get_node(pytree, target_path)
        if node is not None and hasattr(node, "shape") and len(node.shape) > 0:
            fields.append(
                JaxField(path=path, type=str(node_type), shape=tuple(node.shape))
            )

    return fields
