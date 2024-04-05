from dataclasses import dataclass
from typing import Optional


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
