import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.layers.batch_norm import BatchNorm


class Downsample(eqx.Module):
    avg: eqx.nn.AvgPool2d
    conv: eqx.nn.Conv2d
    bn: BatchNorm

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        key: PRNGKeyArray,
    ):
        _, subkey = jax.random.split(key)
        self.avg = eqx.nn.AvgPool2d(stride)
        self.conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            use_bias=False,
            key=subkey,
        )

        self.bn = BatchNorm(out_channels, axis_name="batch")

    def __call__(
        self,
        x: Float[Array, "c_in h w"],
        state: eqx.nn.State,
        *,
        inference: bool = False,
    ) -> tuple[Float[Array, "c_out*e h/s w/s"], eqx.nn.State]:
        x = self.avg(x)
        x = self.conv(x)
        x, state = self.bn(x, state, inference=inference)

        return x, state


class Bottleneck(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: BatchNorm

    conv3: eqx.nn.Conv2d
    bn3: BatchNorm

    downsample: Downsample | None

    avgpool: eqx.nn.AvgPool2d | None

    expansion: int = eqx.field(static=True, default=4)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        key: PRNGKeyArray,
    ):
        _, *subkeys = jax.random.split(key, 4)

        self.conv1 = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, use_bias=False, key=subkeys[0]
        )
        self.bn1 = BatchNorm(out_channels, axis_name="batch")

        self.conv2 = eqx.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            use_bias=False,
            key=subkeys[1],
        )

        self.bn2 = BatchNorm(out_channels, axis_name="batch")

        self.avgpool = eqx.nn.AvgPool2d(stride) if stride > 1 else None

        self.conv3 = eqx.nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            key=subkeys[2],
            use_bias=False,
        )

        self.bn3 = BatchNorm(out_channels * self.expansion, axis_name="batch")

        self.downsample = None

        if stride > 1 or in_channels != out_channels * Bottleneck.expansion:
            key, subkey = jax.random.split(key)
            self.downsample = Downsample(
                in_channels, out_channels * self.expansion, stride=1, key=subkey
            )

    def __call__(self, x: Array, state: eqx.nn.State, inference: bool = False):
        identity = x

        out, state = jax.nn.relu(self.bn1(self.conv1(x), state, inference=inference))
        out = jax.nn.relu(self.bn2(self.conv2(out), state, inference=inference))
        if self.avgpool:
            out = self.avgpool(out)
        out, state = self.bn3(self.conv3(out), state, inference=inference)

        if self.downsample is not None:
            identity, state = self.downsample(x, state)

        out += identity
        out = jax.nn.relu(out)
        return out, state


class AttentionPool2d(eqx.Module):
    positional_embedding: Array

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        key: PRNGKeyArray,
        output_dim: int | None = None,
    ):
        key, *subkeys = jax.random.split(key, 5)
        self.positional_embedding = (
            jax.random.normal(key, (spacial_dim**2 + 1, embed_dim)) / embed_dim**0.5
        )
        self.k_proj = eqx.nn.Linear(embed_dim, embed_dim, key=subkeys[0])
        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, key=subkeys[1])
        self.v_proj = eqx.nn.Linear(embed_dim, embed_dim, key=subkeys[2])
        self.c_proj = eqx.nn.Linear(embed_dim, output_dim or embed_dim, key=subkeys[3])
        self.num_heads = num_heads

    def __call__(self, x: Float[Array, "c h w"]):
        c, h, w = x.shape
        x = jnp.einsum("chw->hwc", x).reshape(h * w, c)
        x = jnp.concatenate([jnp.mean(x, axis=0, keepdims=True), x], axis=0)

        x = x + self.positional_embedding

        # x, _ = F.multi_head_attention_forward(
        #     query=x[:1],
        #     key=x,
        #     value=x,
        #     embed_dim_to_check=x.shape[-1],
        #     num_heads=self.num_heads,
        #     q_proj_weight=self.q_proj.weight,
        #     k_proj_weight=self.k_proj.weight,
        #     v_proj_weight=self.v_proj.weight,
        #     in_proj_weight=None,
        #     in_proj_bias=torch.cat(
        #         [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
        #     ),
        #     bias_k=None,
        #     bias_v=None,
        #     add_zero_attn=False,
        #     dropout_p=0,
        #     out_proj_weight=self.c_proj.weight,
        #     out_proj_bias=self.c_proj.bias,
        #     use_separate_proj_weight=True,
        #     training=self.training,
        #     need_weights=False,
        # )
        return x.squeeze(0)
