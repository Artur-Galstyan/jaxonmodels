from typing import Literal

import jax
from jaxonloader import get_tiny_shakespeare, make
from kira import Kira, Mamba
from kira.generate import generate_text
from kira.model_args import get_kira_args, get_mamba_args
from kira.train import train


def main():
    max_seq_len = 8
    early_stop = 500
    batch_size = 64
    tinyshakespeare = get_tiny_shakespeare()
    train_dataset, test_dataset, vocab_size, encode, decode = tinyshakespeare
    key = jax.random.PRNGKey(100)
    key, subkey = jax.random.split(key)
    train_dataloader, train_index = make(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        key=key,
        jit=True,
    )
    test_dataloader, test_index = make(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        key=subkey,
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
        train_index,
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

    # train_mamba(
    #     train_dataloader,
    #     train_index,
    #     n_dims,
    #     n_embd,
    #     n_layers,
    #     learning_rate,
    #     early_stop,
    #     key,
    # )

    generate_text(kira, max_seq_len, 200, decode, vocab_size)


def train_mamba(
    train_dataloader,
    train_index,
    n_dims,
    n_embd,
    n_layers,
    learning_rate,
    early_stop,
    key,
):
    model_args = get_mamba_args(
        n_embd=n_embd, n_dims=n_dims, n_layers=n_layers, d_state=4
    )
    mamba = Mamba(model_args=model_args, key=key)
    key, subkey = jax.random.split(key)
    mamba = train(
        train_dataloader,
        train_index,
        learning_rate,
        mamba,
        subkey,
        early_stop=early_stop,
    )
    return mamba


def train_kira(
    train_dataloader,
    train_index,
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
    kira_model_args = get_kira_args(
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
    )
    key = jax.random.PRNGKey(kira_model_args.key_seed)
    kira = Kira(
        model_args=kira_model_args,
        key=key,
    )
    key, subkey = jax.random.split(key)
    kira = train(
        train_dataloader,
        train_index,
        learning_rate,
        kira,
        early_stop=early_stop,
        key=subkey,
    )
    return kira


if __name__ == "__main__":
    with jax.checking_leaks():
        main()
