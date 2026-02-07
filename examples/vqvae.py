import clu.metrics as clum  # ty:ignore[unresolved-import]
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf  # ty:ignore[unresolved-import]
import tensorflow_datasets as tfds  # ty:ignore[unresolved-import]
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
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


def reparameterize(mu: Array, logvar: Array, key: PRNGKeyArray) -> Array:
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, shape=mu.shape)
    return mu + std * eps


class Encoder(eqx.Module):
    layers: eqx.nn.Sequential

    mu_layer: eqx.nn.Linear
    logvar_layer: eqx.nn.Linear

    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int, key: PRNGKeyArray
    ):
        key, *subkeys = jax.random.split(key, 10)

        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.Linear(input_dim, hidden_dim, key=subkeys[0]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(hidden_dim, hidden_dim, key=subkeys[1]),
                eqx.nn.Lambda(jax.nn.relu),
            ]
        )

        self.mu_layer = eqx.nn.Linear(hidden_dim, latent_dim, key=subkeys[2])
        self.logvar_layer = eqx.nn.Linear(hidden_dim, latent_dim, key=subkeys[3])

    def __call__(self, x: Float[Array, " input_dim"]) -> tuple[Array, Array]:
        x = self.layers(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        return mu, logvar


class Decoder(eqx.Module):
    layers: eqx.nn.Sequential

    def __init__(
        self, latent_dim: int, hidden_dim: int, output_dim: int, key: PRNGKeyArray
    ):
        key, *subkeys = jax.random.split(key, 10)

        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.Linear(latent_dim, hidden_dim, key=subkeys[0]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(hidden_dim, hidden_dim, key=subkeys[1]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(hidden_dim, output_dim, key=subkeys[2]),
            ]
        )

    def __call__(self, z: Float[Array, "latent_dim"]) -> Array:
        x = self.layers(z)
        return x


class VAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int, key: PRNGKeyArray
    ):
        key, *subkeys = jax.random.split(key, 5)
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, key=subkeys[0])
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, key=subkeys[1])

    def __call__(self, x: Array, key: PRNGKeyArray):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar, key)

        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar


def vae_loss(
    model: VAE, x: Float[Array, "batch_size input_dim"], key: PRNGKeyArray
) -> Array:
    keys = jax.random.split(key, len(x))
    reconstructed_x, mu, logvar = eqx.filter_vmap(model)(x, keys)

    recon_loss = jnp.mean(jnp.sum(jnp.square(x - reconstructed_x), axis=-1))

    kl_loss = -0.5 * jnp.mean(
        jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)
    )

    total_loss = recon_loss + kl_loss
    return total_loss


vae_model = VAE(input_dim=784, hidden_dim=128, latent_dim=8, key=jax.random.key(44))

images, labels = next(iter(train_ds.take(1)))
images = jnp.array(images)
labels = jnp.array(labels)
loss = vae_loss(vae_model, images, key=jax.random.key(42))
print(loss)

learning_rate = 3e-4
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(vae_model, eqx.is_array))


@eqx.filter_jit
def step(
    model: PyTree,
    x: Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
):
    loss, grads = eqx.filter_value_and_grad(vae_loss)(model, x, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state


n_epochs = 500

metrics = LossMetrics.empty()
key = jax.random.key(0)
for epoch in tqdm(range(n_epochs)):
    for images, _ in train_ds:
        images = jnp.array(images)
        key, subkey = jax.random.split(key)
        loss, vae_model, opt_state = step(
            vae_model, images, optimizer, opt_state, subkey
        )
        metrics = metrics.merge(LossMetrics.single_from_model_output(loss=loss))

    print(f"Epoch {epoch}, loss {metrics.compute()}")


eqx.tree_serialise_leaves("mnist_vae", vae_model)

vae_model = eqx.tree_deserialise_leaves("mnist_vae", vae_model)


def visualize_reconstructions(model, images, n=10):
    # Get the first n images
    original = images[:n]

    # Reconstruct images
    key = jax.random.key(0)
    keys = jax.random.split(key, n)
    reconstructed, _, _ = eqx.filter_vmap(model)(original, keys)

    # Reshape for visualization
    original = original.reshape(-1, 28, 28)
    reconstructed = reconstructed.reshape(-1, 28, 28)

    # Plot
    fig, axes = plt.subplots(2, n, figsize=(n, 2))
    for i in range(n):
        axes[0, i].imshow(original[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i], cmap="gray")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("reconstructions.png")
    plt.close()


# Generate random samples from the latent space
def visualize_samples(model, n=10):
    # Sample from the latent space
    key = jax.random.key(42)
    z = jax.random.normal(key, shape=(n, model.encoder.mu_layer.out_features))

    # Decode the samples
    samples = eqx.filter_vmap(model.decoder)(z)

    # Reshape for visualization
    samples = samples.reshape(-1, 28, 28)

    # Plot
    fig, axes = plt.subplots(1, n, figsize=(n, 1))
    for i in range(n):
        axes[i].imshow(samples[i], cmap="gray")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("samples.png")
    plt.close()


# Call these after training
test_images, _ = next(iter(test_ds))
test_images = jnp.array(test_images)
visualize_reconstructions(vae_model, test_images)
visualize_samples(vae_model)
