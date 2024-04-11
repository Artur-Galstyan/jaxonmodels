import functools as ft
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxonloader import JaxonDataLoader
from jaxtyping import Array, Int, PRNGKeyArray, PyTree
from loguru import logger


def train(
    train_dataloader: JaxonDataLoader,
    learning_rate: float,
    model: PyTree,
    key: PRNGKeyArray,
    early_stop: int | None = None,
    wandb_client: Any | None = None,
    log_every: Optional[int] = 100,
) -> PyTree:
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_value = 0

    for i, x in enumerate(train_dataloader):
        x = jnp.array(x)
        x, y = jnp.split(x, 2, axis=1)
        key, subkey = jax.random.split(key)
        model, opt_state, loss_value = step(
            model, opt_state, x, y, optimizer, key=subkey
        )

        if log_every is not None and i % log_every == 0:
            logger.info(f"Loss: {loss_value}")
        if early_stop is not None and i > early_stop:
            break

        i += 1
    logger.info("Finished training")
    logger.info(f"Final loss: {loss_value}")
    return model


def evaluate(
    test_dataloader: JaxonDataLoader,
    test_index: eqx.nn.State,
    model: PyTree,
):
    loss = 0
    for x in test_dataloader:
        x, y = jnp.split(jnp.array(x), 2, axis=1)
        x = jnp.array(x)
        y = jnp.array(y)
        loss += loss_fn(model, x, y, key=None)

    return loss / len(test_dataloader)


@eqx.filter_jit
def loss_fn(
    model: PyTree,
    x: Int[Array, "batch_size max_seq_len n_dims"],
    labels: Int[Array, "batch_size max_seq_len n_dims"],
    key: Optional[PRNGKeyArray],
) -> Array:
    partial_model = ft.partial(model, state=None, key=key)
    logits = eqx.filter_vmap(partial_model)(x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


@eqx.filter_jit
def step(
    model: PyTree,
    opt_state: PyTree,
    x: Array,
    y: Array,
    optimizer: optax.GradientTransformation,
    key: PRNGKeyArray,
) -> tuple[PyTree, PyTree, Any]:
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss
