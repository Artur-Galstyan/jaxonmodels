from typing import Literal

import jax
from jaxonloader import JaxonDataLoader
from jaxonloader.datasets import get_tiny_shakespeare
from jaxonmodels.transformers.kira import Kira
from jaxonmodels.transformers.kira.generate import generate_text
from jaxonmodels.transformers.kira.model_args import KiraModelArgs
from jaxonmodels.transformers.kira.train import train


def main():
    max_seq_len = 8
    early_stop = 100
    batch_size = 64
    tinyshakespeare = get_tiny_shakespeare()
    train_dataset, test_dataset, vocab_size, encode, decode = tinyshakespeare
    key = jax.random.PRNGKey(100)
    key, subkey = jax.random.split(key)
    train_dataloader = JaxonDataLoader(
        train_dataset,
        batch_size=batch_size,
    )
    n_dims = vocab_size
    n_embd = 64  # 384
    learning_rate = 3e-4
    num_heads = 4  # 6
    query_multihead_dim = num_heads
    kv_multihead_dim = 2
    n_layers = 3  # 6
    max_new_tokens = 200  # noqa

    kira = train_kira(
        train_dataloader,
        n_dims,
        n_embd,
        n_layers,
        max_seq_len,
        num_heads,
        query_multihead_dim,
        kv_multihead_dim,
        learning_rate,
        early_stop,
        kv_interpolation_mode="repeat",
    )

    generate_text(kira, max_seq_len, 200, decode, vocab_size)


def train_kira(
    train_dataloader,
    n_dims,
    n_embd,
    n_layers,
    max_seq_len,
    num_heads,
    query_multihead_dim,
    kv_multihead_dim,
    learning_rate,
    early_stop,
    kv_interpolation_mode: Literal["average", "repeat"] = "average",
):
    kira_model_args = KiraModelArgs(
        n_dims=n_dims,
        n_embd=n_embd,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        num_query_heads=query_multihead_dim,
        num_kv_heads=kv_multihead_dim,
        width_size=256,
        depth=4,
        key_seed=0,
        kv_interpolation_mode=kv_interpolation_mode,
        p=0.2,
    )
    key = jax.random.PRNGKey(kira_model_args.key_seed)
    kira = Kira(
        model_args=kira_model_args,
        key=key,
    )
    key, subkey = jax.random.split(key)
    kira = train(
        train_dataloader,
        learning_rate,
        kira,
        early_stop=early_stop,
        key=subkey,
    )
    return kira


if __name__ == "__main__":
    with jax.checking_leaks():
        main()
