import os

import clu.metrics as clum
import equinox as eqx
import flax
import grain.python as grain
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optax
from beartype.typing import Any, SupportsIndex
from jaxonlayers.functions.embedding import sinusoidal_embedding
from jaxonlayers.layers import TransformerEncoder
from jaxtyping import Array, Float, Int, PRNGKeyArray


def setup_mlflow(experiment_name: str = "Text Diffusion"):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    assert tracking_uri is not None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@flax.struct.dataclass
class LossMetrics(clum.Collection):
    loss: clum.Average.from_output("loss")  # ty:ignore[invalid-type-form]


class DataFrameDataSource(grain.RandomAccessDataSource):
    def __init__(self, data_list):
        self._data = data_list

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, record_key: SupportsIndex) -> dict:
        return self._data[record_key]


class TinyShakespearePreprocessor(grain.MapTransform):
    def map(self, element):
        return {
            "input": jnp.array(element["input"]),
            "target": jnp.array(element["target"]),
        }


dataset_path = "shakespeare.txt"
batch_size = 128
sequence_length = 128
num_epochs = 500
learning_rate = 3e-4
embed_dim = 2048
time_embedding_size = 256
timesteps = 500

with open(dataset_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}


def decode(l: list[int]) -> str:
    return "".join([int_to_char[int(i)] for i in l])


full_data_ints = np.array([char_to_int[c] for c in text], dtype=np.int32)
train_cutoff = int(len(full_data_ints) * 0.9)
train_data_ints = full_data_ints[:train_cutoff]

train_sequences = []
for i in range(0, len(train_data_ints) - sequence_length, sequence_length):
    input_chunk = train_data_ints[i : i + sequence_length]
    target_chunk = train_data_ints[i + 1 : i + sequence_length + 1]
    record = {"input": input_chunk, "target": target_chunk}
    train_sequences.append(record)


class Model(eqx.Module):
    word_embeddings_layer: eqx.nn.Embedding
    rope: eqx.nn.RotaryPositionalEmbedding
    time_mlp: eqx.nn.MLP

    encoder: TransformerEncoder

    final_conv: eqx.nn.Conv1d

    embed_dim: int
    time_embedding_size: int

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        time_embedding_size: int,
        n_heads: int = 32,
        *,
        key: PRNGKeyArray,
        dtype: Any,
    ):
        self.time_embedding_size = time_embedding_size
        self.embed_dim = embed_dim

        key, word_embeddings_layer_key = jax.random.split(key)

        self.word_embeddings_layer = eqx.nn.Embedding(
            vocab_size + 1, embed_dim, key=word_embeddings_layer_key, dtype=dtype
        )
        self.rope = eqx.nn.RotaryPositionalEmbedding(
            embedding_size=embed_dim // n_heads, dtype=dtype
        )
        key, encoder_key = jax.random.split(key)

        self.encoder = TransformerEncoder(
            d_model=embed_dim,
            n_heads=n_heads,
            key=encoder_key,
            num_layers=18,
            dtype=dtype,
        )

        key, subkey = jax.random.split(key)
        self.final_conv = eqx.nn.Conv1d(
            embed_dim, vocab_size, kernel_size=1, key=subkey, dtype=dtype
        )

        key, mlp_key = jax.random.split(key)
        self.time_mlp = eqx.nn.MLP(
            in_size=time_embedding_size,
            out_size=embed_dim,
            width_size=time_embedding_size * 4,
            depth=2,
            key=mlp_key,
            dtype=dtype,
        )

    def embed(self, x_t: Int[Array, "seq_len"]) -> Float[Array, "seq_len embed_dim"]:
        embs = eqx.filter_vmap(self.word_embeddings_layer)(x_t)

        seq_len, *_ = x_t.shape
        pos_embs = sinusoidal_embedding(
            jnp.expand_dims(jnp.arange(seq_len), axis=1).astype(jnp.bfloat16),
            self.embed_dim,
            dtype=jnp.bfloat16,
        )

        embs = embs + pos_embs

        mean = jnp.mean(embs, axis=-1, keepdims=True)
        std = jnp.std(embs, axis=-1, keepdims=True) + 1e-5
        return (embs - mean) / std

    def _process_heads(
        self,
        q: Float[Array, "seq_length num_heads qk_size"],
        k: Float[Array, "seq_length num_heads qk_size"],
        v: Float[Array, "seq_length num_heads vo_size"],
    ) -> tuple[
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads qk_size"],
        Float[Array, "seq_length num_heads vo_size"],
    ]:
        query_heads = eqx.filter_vmap(self.rope, in_axes=1, out_axes=1)(q)
        key_heads = eqx.filter_vmap(self.rope, in_axes=1, out_axes=1)(k)
        return query_heads, key_heads, v

    def __call__(
        self,
        masked_sequence: Int[Array, "seq_len "],
        t: Int[Array, ""],
        key: PRNGKeyArray,
    ) -> Float[Array, "seq_len vocab_size"]:
        embeddings = self.embed(masked_sequence)
        time_embeddings = sinusoidal_embedding(
            t, self.time_embedding_size, dtype=jnp.bfloat16
        )
        time = self.time_mlp(time_embeddings)
        key, subkey = jax.random.split(key)

        embeddings += time
        encodings = self.encoder(
            embeddings, key=subkey, process_heads=self._process_heads
        )

        output = self.final_conv(encodings.T)
        return output.T  # this transpose makes it seq_len x vocab_size I think


def q_sample(
    x_start: Int[Array, "seq_len"],
    t: Int[Array, ""],
    T: Int[Array, ""],
    vocab_size: Int[Array, ""],
    key: PRNGKeyArray,
) -> Int[Array, "seq_len"]:
    seq_len, *_ = x_start.shape
    key, subkey = jax.random.split(key)
    mask = jax.random.uniform(subkey, shape=(seq_len,)) > (1 - (t / T))
    corrupted_sequence = jnp.where(mask, vocab_size, x_start)
    assert isinstance(corrupted_sequence, Array)
    return corrupted_sequence


def loss_fn(
    model: Model,
    x_0_batch: Int[Array, "batch seq_len"],
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    batch_size, *_ = x_0_batch.shape
    key, subkey = jax.random.split(key)
    batch_keys = jax.random.split(subkey, batch_size)

    key, t_key = jax.random.split(key)
    t = jax.random.randint(t_key, shape=(batch_size,), minval=0, maxval=timesteps)

    masked_sequences = eqx.filter_vmap(q_sample, in_axes=(0, 0, None, None, 0))(
        x_0_batch,
        t,
        jnp.array(timesteps).astype(jnp.bfloat16),
        jnp.array(vocab_size),
        batch_keys,
    )
    keys = jax.random.split(key, batch_size)
    logits = eqx.filter_vmap(model)(masked_sequences, t, keys)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, x_0_batch)
    return jnp.mean(loss)


@eqx.filter_jit(donate="all")
def train_step(
    model: Model,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    x_0_batch: Int[Array, "batch seq_len"],
    key: PRNGKeyArray,
):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x_0_batch, key)
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


@eqx.filter_jit
def _generate_embeddings_jit(
    model: Model,
    num_samples: int,
    seq_len: int,
    key: PRNGKeyArray,
) -> Int[Array, "num_samples seq_len"]:
    keys = jax.random.split(key, num_samples)

    def single_sample_generation(k):
        x_t_init = jnp.full(shape=(seq_len,), fill_value=vocab_size)

        def scan_fn(carry, t):
            x_t, current_key = carry
            current_key, subkey1, subkey2, subkey3 = jax.random.split(current_key, 4)
            t_array = jnp.array(t).astype(jnp.bfloat16)
            logits = model(x_t, t_array, subkey3)  # seq_len x vocab_size

            sampled_tokens = jax.random.categorical(key=subkey1, logits=logits)

            next_sequence = jax.lax.cond(
                t > 0,
                lambda k: q_sample(
                    sampled_tokens,
                    t - 1,
                    jnp.array(timesteps),
                    jnp.array(vocab_size),
                    key=k,
                ),
                lambda k: sampled_tokens,
                subkey2,
            )

            return (next_sequence, current_key), None

        timesteps_array = jnp.arange(timesteps - 1, -1, -1).astype(jnp.bfloat16)
        (x_final, _), _ = jax.lax.scan(scan_fn, (x_t_init, k), timesteps_array)
        return x_final

    return jax.vmap(single_sample_generation)(keys)


def generate_samples(
    model: Model,
    num_samples: int,
    key: PRNGKeyArray,
) -> list[str]:
    samples = _generate_embeddings_jit(model, num_samples, sequence_length, key)
    texts = []
    for sample in samples:
        text = ""
        for token in sample:
            text += int_to_char[int(token)]
        texts.append(text)

    return texts


def main():

    import jax.sharding as js

    mesh = jax.make_mesh(
        (len(jax.devices()),), axis_names=("batch",), axis_types=(js.AxisType.Auto,)
    )
    data_sharding = js.NamedSharding(
        mesh,
        js.PartitionSpec(
            "batch",
        ),
    )
    model_sharding = js.NamedSharding(mesh, js.PartitionSpec())

    key = jax.random.key(42)
    model_key, train_key, sample_key = jax.random.split(key, 3)

    model = Model(
        vocab_size, embed_dim, time_embedding_size, key=model_key, dtype=jnp.bfloat16
    )

    model_weights = eqx.filter(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(model_weights)
    total = sum(x.size for x in leaves)
    print(f"TOTAL MODEL SIZE: {total}")
    print(f"{total / 1e6:.1f}M")

    print(generate_samples(model, 2, jax.random.key(22)))

    train_source = DataFrameDataSource(train_sequences)
    index_sampler = grain.IndexSampler(
        num_records=len(train_source),
        num_epochs=num_epochs,
        shard_options=grain.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        shuffle=True,
        seed=0,
    )
    loader = grain.DataLoader(
        data_source=train_source,
        operations=[TinyShakespearePreprocessor(), grain.Batch(batch_size=batch_size)],
        sampler=index_sampler,
        worker_count=0,
    )

    steps_per_epoch = len(train_source) // batch_size
    total_steps = num_epochs * steps_per_epoch
    print(f"Dataset size: {len(train_source)} sequences")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")

    warmup_steps = int(total_steps * 0.05)

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-6,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=scheduler),
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

    metrics = LossMetrics.empty()
    tracking_url = os.environ.get("MLFLOW_TRACKING_URI", None)
    assert tracking_url is not None
    mlflow.set_tracking_uri(tracking_url)
    mlflow.set_experiment("TinyShakespeare_Diffusion_DISCRETE")

    with mlflow.start_run():
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("embed_dim", embed_dim)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)

        for step, batch in enumerate(loader):
            x_0_batch = batch["input"]
            x_0_batch = eqx.filter_shard(x_0_batch, data_sharding)
            train_key, step_key = jax.random.split(train_key)

            model, opt_state, loss_val = train_step(
                model, optimizer, opt_state, x_0_batch, step_key
            )

            metrics = metrics.merge(LossMetrics.single_from_model_output(loss=loss_val))

            if (step + 1) % steps_per_epoch == 0:
                epoch = (step + 1) // steps_per_epoch
                avg_loss = metrics.compute()["loss"]
                print(f"Epoch {epoch}/{num_epochs}: Loss = {avg_loss:.6f}")
                mlflow.log_metric("train_loss", float(avg_loss), step=step)

                if epoch % 5 == 0:
                    print(f"--- Generating Samples for Epoch {epoch} ---")
                    sample_key, gen_key = jax.random.split(sample_key)
                    samples = generate_samples(model, num_samples=3, key=gen_key)
                    for idx, s in enumerate(samples):
                        print(f"Sample {idx}: {s}")
                        mlflow.log_text(s, f"samples/epoch_{epoch}_sample_{idx}.txt")

                metrics = LossMetrics.empty()


if __name__ == "__main__":
    main()
