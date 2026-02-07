import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf  # ty:ignore[unresolved-import]
import tensorflow_datasets as tfds  # ty:ignore[unresolved-import]
from clu import metrics  # ty:ignore[unresolved-import]
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from tqdm import tqdm

from jaxonmodels.models.vit import VisionTransformer


class Model(eqx.Module):
    vit: VisionTransformer
    norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray):
        key, subkey, subkey2 = jax.random.split(key, 3)
        self.vit = VisionTransformer(
            input_resolution=32,
            patch_size=2,
            width=192,
            layers=4,
            heads=4,
            output_dim=64,
            key=subkey,
        )
        self.norm = eqx.nn.LayerNorm(shape=(64,), use_bias=True)

        self.dropout = eqx.nn.Dropout(p=0.2)
        self.mlp = eqx.nn.MLP(
            in_size=64,
            width_size=64,
            out_size=10,
            depth=2,
            activation=jax.nn.gelu,
            key=subkey2,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        state: eqx.nn.State | None,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, eqx.nn.State | None]:
        x, state = self.vit(x, state=state, inference=inference)
        x = self.norm(x)
        x = self.dropout(x, inference=inference, key=key)
        x = self.mlp(x)
        return x, state


# Load SVHN dataset
(train_ds, test_ds), ds_info = tfds.load(
    "svhn_cropped",
    split=["train", "test"],
    as_supervised=True,
    with_info=True,
)  # pyright: ignore


# Define preprocessing function
def preprocess_image(image, label):
    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0  # pyright: ignore

    # One-hot encode the labels
    label = tf.one_hot(label, depth=10)

    image = tf.transpose(image, (2, 0, 1))

    return image, label


def augment_image(image, label):
    # NOTE: Ensure your augmentation functions work with the correct image format
    # If image is in channels-first format (C, H, W), transpose back for TF ops
    if image.shape[0] == 3:  # If channels-first
        image = tf.transpose(image, (1, 2, 0))  # Convert to (H, W, C) for TF ops

        # Apply augmentations that expect channels-last format
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # Additional augmentations for SVHN
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

        # Ensure pixel values remain in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        # Convert back to channels-first
        image = tf.transpose(image, (2, 0, 1))
    else:
        # Apply augmentations for channels-last format
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(buffer_size=10000)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(AUTOTUNE)

test_ds = test_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(AUTOTUNE)


train_dataset = tfds.as_numpy(train_ds)
test_dataset = tfds.as_numpy(test_ds)

model, state = eqx.nn.make_with_state(Model)(key=jax.random.key(42))


def loss_fn(
    model: Model,
    x: Float[Array, "b c h w"],
    y: Int[Array, "b 10"],
    state: eqx.nn.State,
    key: PRNGKeyArray | None,
    inference: bool = False,
) -> tuple[Array, tuple[Array, eqx.nn.State]]:
    model_pt = functools.partial(model, inference=inference, key=key)
    logits, state = eqx.filter_vmap(
        model_pt, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(x, state)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, jnp.argmax(y, axis=1)
    )
    return jnp.mean(loss), (logits, state)


@eqx.filter_jit
def step(
    model: PyTree,
    x: Array,
    y: Array,
    optimizer: optax.GradientTransformationExtraArgs,
    opt_state: optax.OptState,
    state: eqx.nn.State,
    key: PRNGKeyArray,
):
    print("JIT")
    (loss_value, (logits, state)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(model, x, y, state, key)
    predictions = jnp.argmax(logits, axis=1)
    labels = jnp.argmax(y, axis=1)
    accuracy = jnp.mean(predictions == labels)

    updates, opt_state = optimizer.update(grads, opt_state, model, value=accuracy)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, logits, state


class TrainMetrics(eqx.Module, metrics.Collection):
    loss: metrics.Average.from_output("loss")  # ty:ignore[invalid-type-form]
    accuracy: metrics.Accuracy


def eval(
    model: Model, test_dataset, state: eqx.nn.State, key: PRNGKeyArray
) -> tuple[TrainMetrics, eqx.nn.State]:
    eval_metrics = TrainMetrics.empty()
    for x, y in test_dataset:
        y = jnp.array(y, dtype=jnp.int32)
        loss, (logits, state) = loss_fn(
            model, x, y, state=state, inference=True, key=None
        )
        eval_metrics = eval_metrics.merge(
            TrainMetrics.single_from_model_output(
                logits=logits, labels=jnp.argmax(y, axis=1), loss=loss
            )
        )

    return eval_metrics, state


train_metrics = TrainMetrics.empty()


def create_learning_rate_schedule(
    base_learning_rate: float,
    num_examples: int,
    batch_size: int,
    num_epochs: int,
    warmup_epochs: int = 5,
):
    steps_per_epoch = num_examples // batch_size
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=base_learning_rate, transition_steps=warmup_steps
    )

    cosine_steps = total_steps - warmup_steps
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_steps
    )

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps]
    )

    return schedule_fn


# Setup optimizer with weight decay and learning rate schedule
n_examples = ds_info.splits["train"].num_examples
n_epochs = 100
warmup_epochs = 5
base_lr = 1e-3
weight_decay = 1e-4

lr_schedule = create_learning_rate_schedule(
    base_learning_rate=base_lr,
    num_examples=n_examples,
    batch_size=BATCH_SIZE,
    num_epochs=n_epochs,
    warmup_epochs=warmup_epochs,
)


# optimizer = optax.sgd(base_lr)
# opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array_like))

optimizer = optax.chain(
    optax.clip(1.0),
    optax.adamw(
        learning_rate=lr_schedule, weight_decay=weight_decay, b1=0.9, b2=0.999, eps=1e-8
    ),
)

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
        model, opt_state, loss, logits, state = step(
            model, x, y, optimizer, opt_state, state, key
        )
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
    eval_metrics, state = eval(model, test_dataset, state, subkey)
    evals = eval_metrics.compute()
    print(
        f"Epoch {epoch}: "
        f"test_loss={evals['loss']:.4f}, "
        f"test_acc={evals['accuracy']:.4f}"
    )
