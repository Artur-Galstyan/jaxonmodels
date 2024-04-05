from dataclasses import dataclass
from typing import Literal


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
