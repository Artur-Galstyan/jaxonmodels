import os
import pathlib
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.tree_util import FlattenedIndexKey, GetAttrKey, KeyPath, SequenceKey
from jaxtyping import Array, PyTree
from pydantic import BaseModel


class ChunkifiedPytreePath(BaseModel):
    path: str


class ChunkifiedStatedictPath(BaseModel):
    path: str


class TorchField(BaseModel):
    path: str
    shape: tuple[int, ...]
    skip: bool = False


class JaxField(BaseModel):
    path: KeyPath
    shape: tuple[int, ...]
    skip: bool = False


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


def _get_stateindex_fields(obj) -> dict:
    state_indices = {}

    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue

        try:
            attr_value = getattr(obj, attr_name)
            if isinstance(attr_value, eqx.nn.StateIndex):
                state_indices[attr_name] = attr_value
        except:  # noqa
            pass

    return state_indices


def get_node(
    tree: PyTree, path: KeyPath, state_indices: dict | None = None
) -> tuple[PyTree | None, dict | None]:
    if tree is None:
        return None, {}
    else:
        if len(path) == 0:
            return tree, state_indices
    f, *_ = path
    if hasattr(tree, "is_stateful"):
        if state_indices is None:
            state_indices = {}
        indices = _get_stateindex_fields(tree)
        for attr_name in indices:
            index: eqx.nn.StateIndex = indices[attr_name]
            assert isinstance(index, eqx.nn.StateIndex)
            state_indices[index.marker] = index
    if isinstance(f, SequenceKey):
        subtree = tree[f.idx]
    elif isinstance(f, GetAttrKey):
        subtree = getattr(tree, f.name)
    elif isinstance(f, FlattenedIndexKey):
        if isinstance(tree, eqx.nn.State):
            assert state_indices is not None
            index = state_indices[f.key]
            subtree = tree.get(index)
        else:
            subtree = None
    else:
        subtree = None
    return get_node(subtree, path[1:], state_indices)


def serialize_pytree(tree: PyTree, name: str):
    eqx.tree_serialise_leaves(name, tree)


def replace_node(
    tree: PyTree, path: KeyPath, new_value: Array, state_indices: dict | None = None
) -> PyTree:
    def where_wrapper(t):
        node, _ = get_node(t, path=path, state_indices=state_indices)
        return node

    node, _ = get_node(tree, path=path, state_indices=state_indices)

    if node is not None and eqx.is_array(node):
        tree = eqx.tree_at(
            where_wrapper,
            tree,
            new_value.reshape(node.shape),
        )
    else:
        print("WARNING: Couldn't find: ", jax.tree_util.keystr(path))
    return tree


def move_running_fields_to_the_end(
    torchfields: list[TorchField], identifier: str = "running_"
):
    i = 0
    total = 0
    while i + total < len(torchfields):
        if identifier in torchfields[i].path:
            field = torchfields.pop(i)
            torchfields.append(field)
            total += 1
        else:
            i += 1
    return torchfields


def state_dict_to_fields(state_dict: dict[str, np.ndarray]) -> list[TorchField]:
    if state_dict is None:
        return []
    fields: list[TorchField] = []
    for key, value in state_dict.items():
        if hasattr(value, "shape") and len(value.shape) > 0:
            fields.append(TorchField(path=key, shape=tuple(value.shape)))
    return fields


def pytree_to_fields(pytree: PyTree) -> tuple[list[JaxField], dict | None]:
    jaxfields = []
    paths = jax.tree.leaves_with_path(pytree)
    i = {}
    for p in paths:
        keys, arr = p
        n, i = get_node(pytree, keys, i)
        if n is not None and eqx.is_array(n):
            jaxfields.append(JaxField(path=keys, shape=n.shape))
    return jaxfields, i


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

    for key in state_dict.keys():
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
    state_indices: dict | None,
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
                    "and that you can mark arrays as skippable. "
                    f"{t.path=} "
                    f"{jax.tree_util.keystr(j.path)=}"
                )
            state_dict_dir = pathlib.Path(chunkified_statedict_path.path) / "state_dict"
            filename = state_dict_dir / t.path
            new_value = jnp.array(np.load(str(filename) + ".npy"))

            n, _ = get_node(pytree, j.path, state_indices)
            assert n is not None, f"Node {j.path} not found"
            assert can_reshape(n.shape, new_value.shape), (
                f"Cannot reshape {n.shape} into {new_value.shape}"
            )

            pytree = replace_node(pytree, j.path, new_value, state_indices)

    return pytree
