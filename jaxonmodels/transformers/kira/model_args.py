import math
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class LLaMAModelArgs:
    vocab_size: int
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    head_dim: int = 128
    max_batch_size: int = 32
    max_seq_len: int = 2048
    head_dim: int

    def __post_init__(self):
        self.head_dim = self.dim // self.n_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads


@dataclass
class MambaModelArgs:
    n_embd: int
    n_dims: int
    n_layers: int
    d_state: int = 16
    expand: int = 2
    dt_rank: int | str = "auto"
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    max_seq_len: int = 8
    d_inner: int | None = None
    use_in_projection_bias: bool = True
    use_conv_bias: bool = True
    use_out_proj_bias: bool = True
    ssm_use_delta_proj_bias: bool = False
    ssm_use_input_proj_bias: bool = False
    key_seed: int = 0

    def __post_init__(self):
        self.d_inner = int(self.expand * self.n_embd)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.n_embd / self.d_state)

        if self.n_dims % self.pad_vocab_size_multiple != 0:
            self.n_dims += (
                self.pad_vocab_size_multiple
                - self.n_dims % self.pad_vocab_size_multiple
            )


@dataclass
class KiraModelArgs:
    n_dims: int
    n_embd: int
    n_layers: int
    max_seq_len: int
    num_heads: int
    num_query_heads: int
    num_kv_heads: int
    width_size: int
    depth: int
    key_seed: int
    p: float
    kv_interpolation_mode: Literal["average", "repeat"]


def get_mamba_args(
    n_embd: int,
    n_dims: int,
    n_layers: int,
    key_seed: int = 0,
    d_state: int = 16,
    expand: int = 2,
    dt_rank: int | str = "auto",
    d_conv: int = 4,
    pad_vocab_size_multiple: int = 8,
    conv_bias: bool = True,
    bias: bool = False,
    max_seq_len: int = 8,
) -> MambaModelArgs:
    return MambaModelArgs(
        n_embd=n_embd,
        n_dims=n_dims,
        n_layers=n_layers,
        d_state=d_state,
        expand=expand,
        dt_rank=dt_rank,
        d_conv=d_conv,
        pad_vocab_size_multiple=pad_vocab_size_multiple,
        conv_bias=conv_bias,
        bias=bias,
        max_seq_len=max_seq_len,
        key_seed=key_seed,
    )


def get_kira_args(
    n_dims: int,
    n_embd: int,
    n_layers: int,
    max_seq_len: int,
    num_heads: int,
    num_query_heads: int,
    num_kv_heads: int,
    width_size: int,
    depth: int,
    key_seed: int = 0,
    kv_interpolation_mode: Literal["average", "repeat"] = "average",
    p: float = 0.2,
) -> KiraModelArgs:
    return KiraModelArgs(
        n_dims=n_dims,
        n_embd=n_embd,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        width_size=width_size,
        depth=depth,
        p=p,
        key_seed=key_seed,
        kv_interpolation_mode=kv_interpolation_mode,
    )
