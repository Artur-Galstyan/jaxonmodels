import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import metrics
from tqdm import tqdm

from jaxonmodels.models.resnet import ResNet, resnet18

(train, test), info = tfds.load(
    "cifar10", split=["train", "test"], with_info=True, as_supervised=True
) # pyright: ignore


def preprocess(
    img: jt.Float[tf.Tensor, "h w c"], label: jt.Int[tf.Tensor, ""]
) -> tuple[jt.Float[tf.Tensor, "h w c"], jt.Int[tf.Tensor, "1 n_classes"]]:
    img = tf.cast(img, tf.float32) / 255.0 # pyright: ignore
    mean = tf.constant([0.4914, 0.4822, 0.4465])
    std = tf.constant([0.2470, 0.2435, 0.2616])
    img = (img - mean) / std # pyright: ignore

    img = tf.transpose(img, perm=[2, 0, 1])

    # label = tf.one_hot(label, depth=10)

    return img, label


def preprocess_train(
    img: jt.Float[tf.Tensor, "h w c"], label: jt.Int[tf.Tensor, ""]
) -> tuple[jt.Float[tf.Tensor, "h w c"], jt.Int[tf.Tensor, "1 n_classes"]]:
    img = tf.pad(img, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, [32, 32, 3])
    img = tf.image.random_flip_left_right(img)  # pyright: ignore

    return preprocess(img, label)


train_dataset = train.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
SHUFFLE_VAL = len(train_dataset) // 1000
BATCH_SIZE = 128
train_dataset = train_dataset.shuffle(SHUFFLE_VAL)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

train_dataset = tfds.as_numpy(train_dataset)
test_dataset = tfds.as_numpy(test_dataset)



def loss_fn(
    resnet: ResNet,
    x: jt.Array,
    y: jt.Array,
    state: eqx.nn.State,
) -> tuple[jt.Array, tuple[jt.Array, eqx.nn.State]]:
    logits, state = eqx.filter_vmap(
        resnet, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(x, state)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss), (logits, state)

# @eqx.filter_jit
def step(
    resnet: jt.PyTree,
    state: eqx.nn.State,
    x: jt.Array,
    y: jt.Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    (loss_value, (logits, state)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(resnet, x, y, state)
    updates, opt_state = optimizer.update(grads, opt_state, resnet)
    resnet = eqx.apply_updates(resnet, updates)
    return resnet, state, opt_state, loss_value, logits



class TrainMetrics(eqx.Module, metrics.Collection):
    loss: metrics.Average.from_output("loss")  # pyright: ignore
    accuracy: metrics.Accuracy


def eval(
    resnet: ResNet, test_dataset, state, key: jt.PRNGKeyArray
) -> TrainMetrics:
    eval_metrics = TrainMetrics.empty()
    for x, y in test_dataset:
        y = jnp.array(y, dtype=jnp.int32)
        loss, (logits, state) = loss_fn(resnet, x, y, state)
        eval_metrics = eval_metrics.merge(
            TrainMetrics.single_from_model_output(
                logits=logits, labels=y, loss=loss
            )
        )

    return eval_metrics


train_metrics = TrainMetrics.empty()

resnet, state = resnet18(key=jax.random.key(0), n_classes=10)

learning_rate = 0.1
weight_decay = 5e-4
optimizer = optax.sgd(learning_rate)

opt_state = optimizer.init(eqx.filter(resnet, eqx.is_inexact_array_like))

key = jax.random.key(99)
n_epochs = 100


for epoch in range(n_epochs):
    batch_count = len(train_dataset)

    pbar = tqdm(enumerate(train_dataset), total=batch_count, desc=f"Epoch {epoch}")
    for i, (x, y) in pbar:
        y = jnp.array(y, dtype=jnp.int32)
        resnet, state, opt_state, loss, logits = step(
            resnet, state, x, y, optimizer, opt_state
        )
        train_metrics = train_metrics.merge(
            TrainMetrics.single_from_model_output(
                logits=logits, labels=y, loss=loss
            )
        )

        vals = train_metrics.compute()
        pbar.set_postfix(
            {"loss": f"{vals['loss']:.4f}", "acc": f"{vals['accuracy']:.4f}"}
        )
    key, subkey = jax.random.split(key)
    eval_metrics = eval(resnet, test_dataset, state, subkey)
    evals = eval_metrics.compute()
    print(
        f"Epoch {epoch}: "
        f"test_loss={evals['loss']:.4f}, "
        f"test_acc={evals['accuracy']:.4f}"
    )
