import math
from dataclasses import dataclass


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
