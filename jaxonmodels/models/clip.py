import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

import jaxonmodels.functions as F
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
        self.avg = eqx.nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
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
        *,
        axis_name: str = "batch",
        key: PRNGKeyArray,
    ):
        _, *subkeys = jax.random.split(key, 4)

        self.conv1 = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, use_bias=False, key=subkeys[0]
        )
        self.bn1 = BatchNorm(out_channels, axis_name=axis_name)

        self.conv2 = eqx.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=subkeys[1],
        )

        self.bn2 = BatchNorm(out_channels, axis_name=axis_name)

        self.avgpool = (
            eqx.nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else None
        )

        self.conv3 = eqx.nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            key=subkeys[2],
            use_bias=False,
        )

        self.bn3 = BatchNorm(out_channels * self.expansion, axis_name=axis_name)

        self.downsample = None

        if stride > 1 or in_channels != out_channels * Bottleneck.expansion:
            key, subkey = jax.random.split(key)
            self.downsample = Downsample(
                in_channels, out_channels * self.expansion, stride=stride, key=subkey
            )

    def __call__(self, x: Array, state: eqx.nn.State, inference: bool = False):
        identity = x

        out, state = self.bn1(self.conv1(x), state, inference=inference)
        out = jax.nn.relu(out)

        out, state = self.bn2(self.conv2(out), state, inference=inference)
        out = jax.nn.relu(out)

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

    k_proj: eqx.nn.Linear
    q_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    c_proj: eqx.nn.Linear

    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int | None = None,
        *,
        key: PRNGKeyArray,
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

    def __call__(self, x: Float[Array, "c h w"], inference: bool = False):
        c, h, w = x.shape
        x = jnp.einsum("chw->hwc", x).reshape(h * w, c)
        x = jnp.concatenate([jnp.mean(x, axis=0, keepdims=True), x], axis=0)

        x = x + self.positional_embedding

        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=jnp.concatenate(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=not inference,
            need_weights=False,
        )
        return x.squeeze(0)


class ResNetBlock(eqx.Module):
    blocks: list[Bottleneck]

    def __call__(self, x, state, *, inference: bool = False):
        for block in self.blocks:
            x, state = block(x, state, inference=inference)
        return x, state


class ModifiedResNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: BatchNorm

    conv2: eqx.nn.Conv2d
    bn2: BatchNorm

    conv3: eqx.nn.Conv2d
    bn3: BatchNorm

    avgpool: eqx.nn.AvgPool2d

    layer1: ResNetBlock
    layer2: ResNetBlock
    layer3: ResNetBlock
    layer4: ResNetBlock

    attnpool: AttentionPool2d

    _inplanes: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    input_resolution: int = eqx.field(static=True)

    def __init__(
        self,
        layers,
        output_dim,
        heads,
        input_resolution=224,
        width=64,
        *,
        key: PRNGKeyArray,
        axis_name: str = "batch",
    ):
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        key, *subkeys = jax.random.split(key, 7)

        # the 3-layer stem
        self.conv1 = eqx.nn.Conv2d(
            3,
            width // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=False,
            key=subkeys[0],
        )
        self.bn1 = BatchNorm(width // 2, axis_name=axis_name)

        self.conv2 = eqx.nn.Conv2d(
            width // 2,
            width // 2,
            kernel_size=3,
            padding=1,
            use_bias=False,
            key=subkeys[1],
        )
        self.bn2 = BatchNorm(width // 2, axis_name=axis_name)

        self.conv3 = eqx.nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, use_bias=False, key=subkeys[2]
        )
        self.bn3 = BatchNorm(width, axis_name=axis_name)
        self.avgpool = eqx.nn.AvgPool2d(2, stride=2)

        key, *subkeys = jax.random.split(key, 7)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0], key=subkeys[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, key=subkeys[1])
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, key=subkeys[2])
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, key=subkeys[3])

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim, key=subkeys[4]
        )

    def _make_layer(self, planes, blocks, stride=1, *, key: PRNGKeyArray):
        key, *subkeys = jax.random.split(key, blocks + 1)
        layers = [Bottleneck(self._inplanes, planes, stride, key=subkeys[0])]

        self._inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, stride=1, key=subkeys[i]))

        return ResNetBlock(layers)

    def __call__(
        self,
        x: Float[Array, "c h w"],
        *,
        state: eqx.nn.State | None = None,
        inference: bool = False,
    ):
        assert state is not None, "Must give state for ResNet"

        def stem(x, state, inference):
            x, state = self.bn1(self.conv1(x), state, inference=inference)
            x = jax.nn.relu(x)
            x, state = self.bn2(self.conv2(x), state, inference=inference)
            x = jax.nn.relu(x)
            x, state = self.bn3(self.conv3(x), state, inference=inference)
            x = jax.nn.relu(x)
            x = self.avgpool(x)
            return x, state

        x, state = stem(x, state, inference)
        x, state = self.layer1(x, state)
        x, state = self.layer2(x, state)
        x, state = self.layer3(x, state)
        x, state = self.layer4(x, state)
        x = self.attnpool(x)

        return x, state


class ResidualAttentionBlock(eqx.Module):
    attn: eqx.nn.MultiheadAttention
    ln_1: eqx.nn.LayerNorm
    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    ln_2: eqx.nn.LayerNorm

    def __init__(
        self,
        d_model: int,
        n_head: int,
        *,
        key: PRNGKeyArray,
    ):
        key, *subkeys = jax.random.split(key, 5)
        self.attn = eqx.nn.MultiheadAttention(n_head, d_model, key=subkeys[0])
        self.ln_1 = eqx.nn.LayerNorm(d_model)
        self.c_fc = eqx.nn.Linear(d_model, d_model * 4, key=subkeys[1])
        self.c_proj = eqx.nn.Linear(d_model * 4, d_model, key=subkeys[2])
        self.ln_2 = eqx.nn.LayerNorm(d_model)

    def _attention(self, x: Array, attn_mask: Array | None = None):
        return self.attn(x, x, x, mask=attn_mask)

    def __call__(self, x: Array, attn_mask: Array | None = None):
        x = eqx.filter_vmap(self.ln_1)(x)
        x = x + self._attention(x, attn_mask)
        x = eqx.filter_vmap(self.c_fc)(x)
        x = jax.nn.gelu(x)
        x = eqx.filter_vmap(self.c_proj)(x)
        x = x + eqx.filter_vmap(self.ln_2)(x)
        return x


class Transformer(eqx.Module):
    resblocks: list[ResidualAttentionBlock]

    def __init__(self, width: int, layers: int, heads: int, *, key: PRNGKeyArray):
        key, *subkeys = jax.random.split(key, layers + 1)
        self.resblocks = [
            ResidualAttentionBlock(width, heads, key=subkeys[i]) for i in range(layers)
        ]

    def __call__(
        self, x: Float[Array, "seq_len embed_dim"], attn_mask: Array | None = None
    ):
        for resblock in self.resblocks:
            x = resblock(x, attn_mask=attn_mask)

        return x


class VisionTransformer(eqx.Module):
    conv1: eqx.nn.Conv2d
    class_embedding: Array
    positional_embedding: Array
    ln_pre: eqx.nn.LayerNorm
    transformer: Transformer
    ln_post: eqx.nn.LayerNorm
    proj: Array

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        *,
        key: PRNGKeyArray,
    ):
        key, *subkeys = jax.random.split(key, 6)
        self.conv1 = eqx.nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            use_bias=False,
            key=subkeys[0],
        )

        scale = width**-0.5
        self.class_embedding = jax.random.normal(subkeys[1], (width,)) * scale
        self.positional_embedding = scale * jax.random.normal(
            subkeys[2], ((input_resolution // patch_size) ** 2 + 1, width)
        )

        self.ln_pre = eqx.nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, key=subkeys[3])

        self.ln_post = eqx.nn.LayerNorm(width)
        self.proj = scale * jax.random.normal(subkeys[4], (width, output_dim))

    def __call__(
        self,
        x: Float[Array, "c h w"],
        *,
        state: eqx.nn.State | None = None,
        inference: bool = False,
    ):
        x = self.conv1(x)  # shape = [channel, grid, grid]
        x = x.reshape(x.shape[0], -1)  # shape = [channel, grid ** 2]
        x = jnp.transpose(x)  # shape = [*, grid ** 2, width]

        x = jnp.concatenate(
            [
                self.class_embedding.reshape(1, -1),
                x,
            ],
        )
        x = x + self.positional_embedding
        x = eqx.filter_vmap(self.ln_pre)(x)
        x = self.transformer(x)
        x = self.ln_post(x[0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, None


class CLIP(eqx.Module):
    visual: ModifiedResNet | VisionTransformer
    transformer: Transformer
    token_embedding: eqx.nn.Embedding
    positional_embedding: Array
    ln_final: eqx.nn.LayerNorm
    text_projection: Array
    logit_scale: Array

    context_length: int = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: tuple[int, int, int, int] | int,
        vision_width: int,
        vision_patch_size: int | None,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        *,
        key: PRNGKeyArray | None = None,
    ):
        self.context_length = context_length
        if key is None:
            # use default key
            key = jax.random.key(42)
        key, *subkeys = jax.random.split(key, 10)

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                key=subkeys[0],
            )
        else:
            vision_heads = vision_width // 64
            assert vision_patch_size is not None
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                key=subkeys[0],
            )
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            key=subkeys[1],
        )
        self.token_embedding = eqx.nn.Embedding(
            weight=jax.random.normal(
                key=subkeys[2], shape=(vocab_size, transformer_width)
            )
            * 0.02
        )
        self.positional_embedding = (
            jax.random.normal(key=subkeys[3], shape=(context_length, transformer_width))
            * 0.01
        )
        self.ln_final = eqx.nn.LayerNorm(transformer_width)

        self.text_projection = jnp.empty((transformer_width, embed_dim))
        self.logit_scale = jnp.ones([]) * jnp.log(1 / 0.07)

    def initialize_parameters(self, key: PRNGKeyArray):
        # todo
        pass

    def encode_image(self, image, state: eqx.nn.State | None, inference: bool = True):
        return self.visual(image, state=state, inference=inference)

    def encode_text(self, text: Int[Array, "77"], attn_mask: Array):
        x = eqx.filter_vmap(self.token_embedding)(text)  # [n_ctx, d_model]

        x = x + self.positional_embedding
        x = self.transformer(x, attn_mask=attn_mask)
        x = eqx.filter_vmap(self.ln_final)(x)

        x = x[jnp.argmax(text, axis=-1)] @ self.text_projection

        return x

    def __call__(
        self,
        image: Float[Array, "c h w"],
        text: Int[Array, "1 77"],
        state: eqx.nn.State | None,
        attn_mask: Array | None = None,
    ):
        image_features, state = self.encode_image(image, state)
        text = text.reshape(-1)
        if attn_mask is None:
            attn_mask = F.build_attention_mask(self.context_length)
        text_features = self.encode_text(text, attn_mask)

        image_features = jnp.linalg.norm(image_features / image_features, keepdims=True)
        text_features = jnp.linalg.norm(text_features / text_features, keepdims=True)

        logit_scale = jnp.exp(self.logit_scale)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text, state
