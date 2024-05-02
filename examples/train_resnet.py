import functools as ft
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from datasets import load_dataset
from jaxonloader import DataTargetDataset, JaxonDataLoader
from jaxonmodels.vision.resnet import ResNet, resnet18
from jaxtyping import Array, PyTree
from tqdm import tqdm


named_batch_vmap = ft.partial(
    eqx.filter_vmap, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
)

dataset = load_dataset("cifar10", split="train")
dataset = dataset.to_iterable_dataset()  # pyright: ignore


def augment(img, label):
    """Performs data augmentation."""
    image = img
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = np.array(image, dtype=np.float32)
    image = image.transpose(2, 0, 1)
    image = image / 255.0
    return image, label


if not os.path.exists("cifar10_data.npy") and not os.path.exists("cifar10_labels.npy"):
    rows = []
    labels = []
    for img in tqdm(dataset):
        img, label = augment(**img)
        rows.append(img)
        labels.append(label)

    rows = np.array(rows)
    labels = np.array(labels)

    np.save("cifar10_data.npy", rows)
    np.save("cifar10_labels.npy", labels)
else:
    rows = np.load("cifar10_data.npy")
    labels = np.load("cifar10_labels.npy")

print(rows.shape, labels.shape)

train_test_split = 0.8
split_index = int(len(rows) * train_test_split)
train_rows, test_rows = rows[:split_index], rows[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

train_dataset = DataTargetDataset(train_rows, train_labels)
test_dataset = DataTargetDataset(test_rows, test_labels)

train_loader = JaxonDataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = JaxonDataLoader(test_dataset, batch_size=64, shuffle=False)


def loss_fn(
    resnet: ResNet, state: eqx.nn.State, x: Array, y: Array
) -> tuple[Array, eqx.nn.State]:
    out, state = named_batch_vmap(resnet)(x, state)
    return optax.softmax_cross_entropy_with_integer_labels(out, y).mean(), state


@eqx.filter_jit
def step(
    model: PyTree,
    state: eqx.nn.State,
    x: Array,
    y: Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    (loss, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, state, x, y
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, state


N_EPOCHS = 50
PEAK_LR = 0.12
lr_schedule = optax.linear_onecycle_schedule(N_EPOCHS * len(train_loader), PEAK_LR)


def eval(model: PyTree, state: eqx.nn.State, test_dataloader: JaxonDataLoader):
    top_3_accuracy = []
    avg_accuracy = []
    eval_model = eqx.nn.inference_mode(model)
    for x, y in test_dataloader:
        x, y = jnp.array(x), jnp.array(y)
        out, state = named_batch_vmap(eval_model)(x, state)
        accuracy = jnp.mean(jnp.argmax(out, axis=1) == y)
        avg_accuracy.append(accuracy)
        top3_indices = jnp.argsort(out, axis=1)[:, -3:]
        is_target_in_top3 = jnp.array([y[i] in top3_indices[i] for i in range(len(y))])
        top_3_accuracy.append(jnp.mean(is_target_in_top3))
    print(f"Top 3 accuracy: {jnp.mean(jnp.array(top_3_accuracy))}")
    return jnp.mean(jnp.array(avg_accuracy)), state


model, state = resnet18(num_classes=10, key=jax.random.PRNGKey(42))
optimizer = optax.adamw(lr_schedule)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

for epoch in range(N_EPOCHS):
    avg_loss = []

    for x, y in tqdm(train_loader, ascii=True):
        x, y = jnp.array(x), jnp.array(y)
        model, opt_state, loss, state = step(model, state, x, y, optimizer, opt_state)
        avg_loss.append(loss)

    accuracy, state = eval(model, state, test_loader)
    print(f"Epoch {epoch}, loss {jnp.mean(jnp.array(avg_loss))}, accuracy {accuracy}")

# Save the model
eqx.tree_serialise_leaves("resnet18_cifar10.eqx", model)
