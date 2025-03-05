import functools as ft
import os
import pathlib
import re
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array, PyTree
from pydantic import BaseModel
from tqdm import tqdm


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


def serialize_pytree_chunks(tree: PyTree, paths: list[ChunkifiedPytreePath], name: str):
    """
    Reassemble a JAX PyTree from chunked files and serialize it.

    Args:
        tree (PyTree): The original JAX PyTree structure.
        paths (list[ChunkifiedPytreePath]): List of paths to the chunked files.
        name (str): Name of the output serialized file.
    """
    for chunk_path in tqdm(paths):
        array = np.load(chunk_path.path)
        tree = replace_node(tree, chunk_path.path.split(".")[1:-1], array)

    identity = lambda *args, **kwargs: tree
    model, state = eqx.nn.make_with_state(identity)()
    eqx.tree_serialise_leaves(name, (model, state))


def replace_node(tree: PyTree, targets: list[str], new_value: Array) -> PyTree:
    """
    Replace a node in the PyTree with a new value.

    Args:
        tree (PyTree): The PyTree to modify.
        targets (list[str]): Path to the target node.
        new_value (Array): The new value to insert.

    Returns:
        PyTree: The modified PyTree.

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
    new_weight = jax.numpy.ones((20, 10))
    updated_model = replace_node(
                        model,
                        ['layer1', 'weight'],
                        new_weight
                    )
    assert (updated_model.layer1.weight == new_weight).all()
    ```
    """
    where = ft.partial(get_node, targets=targets)
    node = where(tree)

    if node is not None and hasattr(node, "shape"):
        tree = eqx.tree_at(
            where,
            tree,
            new_value.reshape(node.shape),
        )
    else:
        print("Couldn't find: ", targets)
    return tree


def state_dict_to_fields(state_dict: dict[str, np.ndarray]) -> list[TorchField]:
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


def chunkify_state_dict(
    state_dict: dict[str, np.ndarray], target_path: str
) -> ChunkifiedStatedictPath:
    """
    Convert a PyTorch state dict into chunked files and save them to the specified path.

    Args:
        state_dict (dict[str, np.ndarray]): The PyTorch state dict to be chunked.
        target_path (str): The directory where chunked files will be saved.

    Returns:
        ChunkifiedStatedictPath: A path to the chunked files
    """

    for key in tqdm(state_dict.keys()):
        if not hasattr(state_dict[key], "shape"):
            continue
        path = pathlib.Path(target_path) / "state_dict"

        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path / key, state_dict[key])

    return ChunkifiedStatedictPath(path=str(pathlib.Path(target_path)))


def convert(
    state_dict: dict[str, torch.Tensor],
    pytree: PyTree,
    jaxfields: list[JaxField],
    torchfields: list[TorchField],
) -> PyTree:
    state_dict_np: dict[str, np.ndarray] = {
        k: state_dict[k].detach().numpy() for k in state_dict
    }

    if len(torchfields) != len(jaxfields):
        raise ValueError(
            f"Length of state_dict ({len(torchfields)}) "
            f"!= length of pytree ({len(jaxfields)})"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        chunkified_statedict_path = chunkify_state_dict(state_dict_np, tmpdir)
        del state_dict_np, state_dict
        for t, j in zip(torchfields, jaxfields):
            if not can_reshape(t.shape, j.shape):
                raise ValueError(
                    f"Cannot reshape {t.shape} "
                    f"into shape {j.shape}. "
                    "Note that the order of the fields matters "
                    "and that you can mark arrays as skippable"
                )
            state_dict_dir = pathlib.Path(chunkified_statedict_path.path) / "state_dict"
            filename = state_dict_dir / t.path
            new_value = jnp.array(np.load(str(filename) + ".npy"))
            targets = j.path.split(".")[1:]

            n = get_node(pytree, targets)
            assert n is not None, f"Node {targets} not found"
            assert can_reshape(
                n.shape, new_value.shape
            ), f"Cannot reshape {n.shape} into {new_value.shape}"

            pytree = replace_node(
                pytree,
                targets,
                new_value,
            )

    return pytree
