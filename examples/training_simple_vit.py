import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import metrics
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from tqdm import tqdm

from jaxonmodels.models.vit import VisionTransformer

tf.config.set_visible_devices([], "GPU")


class Model(eqx.Module):
    vit: VisionTransformer
    norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray):
        key, subkey, subkey2 = jax.random.split(key, 3)
        self.vit = VisionTransformer(
            input_resolution=32,
            patch_size=2,  # Small patches capture fine details
            width=768,  # Large embedding dimension for better capacity
            layers=8,  # Deep model for hierarchical feature learning
            heads=12,  # Multiple attention heads (64 dims per head)
            output_dim=384,
            key=subkey,
        )

        self.norm = eqx.nn.LayerNorm(shape=(384,), use_bias=True)

        self.dropout = eqx.nn.Dropout(p=0.1)

        self.mlp = eqx.nn.MLP(
            in_size=384,
            width_size=512,
            out_size=10,
            depth=2,
            activation=jax.nn.gelu,
            key=subkey2,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> jnp.ndarray:
        x = self.vit(x)
        x = self.norm(x)
        x = self.dropout(x, inference=inference, key=key)
        x = self.mlp(x)
        return x


# Set random seed for reproducibility
tf.random.set_seed(42)

# Load SVHN dataset
(train_ds, test_ds), ds_info = tfds.load(
    "svhn_cropped",
    split=["train", "test"],
    as_supervised=True,
    with_info=True,
)  # pyright: ignore


# Print dataset info
print("Dataset information:")
print(f"Number of training examples: {ds_info.splits['train'].num_examples}")
print(f"Number of test examples: {ds_info.splits['test'].num_examples}")


# Define preprocessing function
def preprocess_image(image, label):
    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0  # pyright: ignore

    # One-hot encode the labels
    label = tf.one_hot(label, depth=10)

    image = tf.transpose(image, (2, 0, 1))

    return image, label


def augment_image(image, label):
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.2)

    # Random contrast adjustment
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Ensure pixel values remain in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
train_ds_augmented = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(buffer_size=10000)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(AUTOTUNE)

test_ds = test_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(AUTOTUNE)


train_dataset = tfds.as_numpy(train_ds)
test_dataset = tfds.as_numpy(test_ds)

model = Model(key=jax.random.key(42))


def loss_fn(
    model: Model,
    x: Float[Array, "b c h w"],
    y: Int[Array, "b 10"],
    key: PRNGKeyArray | None,
    inference: bool = False,
) -> tuple[Array, Array]:
    model = functools.partial(model, inference=inference, key=key)  # pyright: ignore
    logits = eqx.filter_vmap(model)(x)
    loss = optax.softmax_cross_entropy(logits, y)
    return jnp.mean(loss), logits


@eqx.filter_jit
def step(
    model: PyTree,
    x: Array,
    y: Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
):
    (loss_value, (logits)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, x, y, key
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, logits


class TrainMetrics(eqx.Module, metrics.Collection):
    loss: metrics.Average.from_output("loss")  # pyright: ignore
    accuracy: metrics.Accuracy


def eval(model: Model, test_dataset, key: PRNGKeyArray) -> TrainMetrics:
    eval_metrics = TrainMetrics.empty()
    for x, y in test_dataset:
        y = jnp.array(y, dtype=jnp.int32)
        loss, (logits) = loss_fn(model, x, y, inference=True, key=None)
        eval_metrics = eval_metrics.merge(
            TrainMetrics.single_from_model_output(
                logits=logits, labels=jnp.argmax(y, axis=1), loss=loss
            )
        )

    return eval_metrics


train_metrics = TrainMetrics.empty()

learning_rate = 0.1
# weight_decay = 5e-4
optimizer = optax.adam(learning_rate)

opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

key = jax.random.key(99)
n_epochs = 100


for epoch in range(n_epochs):
    batch_count = len(train_dataset)

    pbar = tqdm(enumerate(train_dataset), total=batch_count, desc=f"Epoch {epoch}")
    for i, (x, y) in pbar:
        x = jnp.array(x)
        y = jnp.array(y, dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        model, opt_state, loss, logits = step(model, x, y, optimizer, opt_state, key)
        train_metrics = train_metrics.merge(
            TrainMetrics.single_from_model_output(
                logits=logits, labels=jnp.argmax(y, axis=1), loss=loss
            )
        )

        vals = train_metrics.compute()
        pbar.set_postfix(
            {"loss": f"{vals['loss']:.4f}", "acc": f"{vals['accuracy']:.4f}"}
        )
    key, subkey = jax.random.split(key)
    eval_metrics = eval(model, test_dataset, subkey)
    evals = eval_metrics.compute()
    print(
        f"Epoch {epoch}: "
        f"test_loss={evals['loss']:.4f}, "
        f"test_acc={evals['accuracy']:.4f}"
    )
