from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from jaxonmodels.layers.selective_state_space_models import SelectiveStateSpaceModel
from jaxonmodels.ssm.mamba.model_args import MambaModelArgs


class MambaBlock(eqx.Module):
    in_proj: eqx.nn.Linear
    conv1d: eqx.nn.Conv1d
    ssm: SelectiveStateSpaceModel
    out_proj: eqx.nn.Linear

    def __init__(
        self,
        n_embd: int,
        d_inner: int,
        dt_rank: int,
        d_conv: int,
        use_in_projection_bias: bool = True,
        use_conv_bias: bool = True,
        use_out_proj_bias: bool = True,
        ssm_use_delta_proj_bias: bool = False,
        ssm_use_input_proj_bias: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        (
            key,
            linear_key,
            conv1d_key,
            ssm_key,
            out_proj_key,
        ) = jax.random.split(key, 5)

        self.in_proj = eqx.nn.Linear(
            n_embd,
            d_inner * 2,
            use_bias=use_in_projection_bias,
            key=linear_key,
        )

        self.conv1d = eqx.nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            use_bias=use_conv_bias,
            groups=d_inner,
            padding=d_conv - 1,
            key=conv1d_key,
        )
        self.ssm = SelectiveStateSpaceModel(
            d_inner=d_inner,
            dt_rank=dt_rank,
            d_state=d_inner,
            use_delta_proj_bias=ssm_use_delta_proj_bias,
            use_input_proj_bias=ssm_use_input_proj_bias,
            key=ssm_key,
        )
        self.out_proj = eqx.nn.Linear(
            d_inner,
            n_embd,
            use_bias=use_out_proj_bias,
            key=out_proj_key,
        )

    def __call__(self, x: Array):
        seq_len, d = x.shape
        x_and_res = jax.vmap(self.in_proj)(x)

        (x, res) = jnp.split(x_and_res, 2, axis=-1)
        x = jnp.transpose(x)
        x = self.conv1d(x)[:, :seq_len]
        x = jnp.transpose(x)
        x = jax.nn.silu(x)

        y = self.ssm(x)
        y = y * jax.nn.silu(res)

        output = jax.vmap(self.out_proj)(y)
        return output


class ResidualBlock(eqx.Module):
    mamba_block: MambaBlock
    rns_norm: eqx.nn.RMSNorm

    def __init__(
        self,
        n_embd: int,
        d_inner: int,
        dt_rank: int,
        d_conv: int,
        use_in_projection_bias: bool = True,
        use_conv_bias: bool = True,
        use_out_proj_bias: bool = True,
        ssm_use_delta_proj_bias: bool = False,
        ssm_use_input_proj_bias: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.mamba_block = MambaBlock(
            n_embd=n_embd,
            d_inner=d_inner,
            dt_rank=dt_rank,
            d_conv=d_conv,
            use_in_projection_bias=use_in_projection_bias,
            use_conv_bias=use_conv_bias,
            use_out_proj_bias=use_out_proj_bias,
            ssm_use_delta_proj_bias=ssm_use_delta_proj_bias,
            ssm_use_input_proj_bias=ssm_use_input_proj_bias,
            key=key,
        )
        self.rns_norm = eqx.nn.RMSNorm(n_embd)

    def __call__(
        self, x: Float[Array, "seq_len n_embd"], *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        return self.mamba_block(jax.vmap(self.rns_norm)(x)) + x


class Mamba(eqx.Module):
    model_args: MambaModelArgs = eqx.field(static=True)

    layers: eqx.nn.Sequential
    normalization: eqx.nn.RMSNorm
    shared_emb_lm_head: eqx.nn.Shared

    def __init__(
        self,
        model_args: MambaModelArgs,
        *,
        key: PRNGKeyArray,
    ):
        self.model_args = model_args
        key, *subkeys = jax.random.split(key, model_args.n_layers + 3)
        assert model_args.d_inner is not None and isinstance(model_args.d_inner, int)
        assert model_args.dt_rank is not None and isinstance(model_args.dt_rank, int)
        self.layers = eqx.nn.Sequential(
            [
                ResidualBlock(
                    n_embd=model_args.n_embd,
                    d_inner=model_args.d_inner,
                    dt_rank=model_args.dt_rank,
                    d_conv=model_args.d_conv,
                    use_in_projection_bias=model_args.use_in_projection_bias,
                    use_conv_bias=model_args.use_conv_bias,
                    use_out_proj_bias=model_args.use_out_proj_bias,
                    ssm_use_delta_proj_bias=model_args.ssm_use_delta_proj_bias,
                    ssm_use_input_proj_bias=model_args.ssm_use_input_proj_bias,
                    key=subkeys[i],
                )
                for i in range(model_args.n_layers)
            ],
        )
        self.normalization = eqx.nn.RMSNorm(model_args.n_embd)

        embedding = eqx.nn.Embedding(
            model_args.n_dims, model_args.n_embd, key=subkeys[model_args.n_layers]
        )
        lm_head = eqx.nn.Linear(
            model_args.n_embd,
            model_args.n_dims,
            use_bias=False,
            key=subkeys[model_args.n_layers + 1],
        )
        where = lambda embed_and_lin: embed_and_lin[1].weight
        get = lambda embed_and_lin: embed_and_lin[0].weight
        self.shared_emb_lm_head = eqx.nn.Shared(
            (embedding, lm_head), where=where, get=get
        )

    def __call__(
        self,
        x: Int[Array, " seq_len"],
        *,
        state: eqx.nn.State | None = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq_len n_dims"]:
        embedding, linear = self.shared_emb_lm_head()
        x = jax.vmap(embedding)(x)

        x = self.layers(x)
        x = jax.vmap(self.normalization)(x)
        logits = jax.vmap(linear)(x)
        return logits
