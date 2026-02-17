import clu.metrics as clum
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from jaxtyping import Array, PRNGKeyArray, PyTree
from tqdm import tqdm


class LossMetrics(eqx.Module, clum.Collection):
    loss: clum.Average.from_output("loss")  # ty:ignore[invalid-type-form]


tf.random.set_seed(42)
np.random.seed(42)

(train_ds, test_ds), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)  # pyright: ignore


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, (-1,))
    return image, label


BATCH_SIZE = 128

train_ds = train_ds.map(preprocess)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(ds_info.splits["train"].num_examples)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(preprocess)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


class Autoencoder(eqx.Module):
    encoder: eqx.nn.Sequential
    decoder: eqx.nn.Sequential

    def __init__(self, dim: int, hidden_dim: int, z_dim: int, key: PRNGKeyArray):
        key, *subkeys = jax.random.split(key, 10)
        self.encoder = eqx.nn.Sequential(
            [
                eqx.nn.Linear(dim, hidden_dim, key=subkeys[0]),
                eqx.nn.Lambda(fn=jax.nn.relu),
                eqx.nn.Linear(hidden_dim, z_dim, key=subkeys[2]),
            ]
        )

        self.decoder = eqx.nn.Sequential(
            [
                eqx.nn.Linear(z_dim, hidden_dim, key=subkeys[2]),
                eqx.nn.Lambda(fn=jax.nn.relu),
                eqx.nn.Linear(hidden_dim, dim, key=subkeys[2]),
            ]
        )

    def __call__(self, x: Array) -> tuple[Array, Array]:
        z = self.encoder(x)
        o = self.decoder(z)
        return o, z


def loss_fn(model, x):
    outputs, _ = eqx.filter_vmap(model)(x)
    return jnp.mean((outputs - x) ** 2)


@eqx.filter_jit
def step(
    model: PyTree,
    x: Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state


model = Autoencoder(dim=784, hidden_dim=256, z_dim=64, key=jax.random.key(22))
learning_rate = 3e-4
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
n_epochs = 20

metrics = LossMetrics.empty()

epoch_bar = tqdm(range(n_epochs))
for epoch in epoch_bar:
    for images, _ in train_ds:
        images = jnp.array(images)
        loss, model, opt_state = step(model, images, optimizer, opt_state)
        metrics = metrics.merge(LossMetrics.single_from_model_output(loss=loss))

    loss_value = metrics.compute()
    epoch_bar.set_postfix_str(f"loss: {loss_value}")


images, labels = next(iter(train_ds.take(1)))
images = jnp.array(images[0])
labels = jnp.array(labels[0])

output, z = model(images)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(images.reshape(28, 28))
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(output.reshape(28, 28))
plt.title("Reconstructed Image")
plt.tight_layout()
plt.show()
