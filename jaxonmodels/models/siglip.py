import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array, Float, PRNGKeyArray

from jaxonmodels.functions.attention import multi_head_attention_forward
from jaxonmodels.functions.utils import default_floating_dtype


class SiglipVisionEmbeddings(eqx.Module):
    patch_embedding: eqx.nn.Conv2d
    position_embedding: eqx.nn.Embedding

    num_channels: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    image_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    num_positions: int = eqx.field(static=True)

    def __init__(
        self,
        num_channels: int,
        embed_dim: int,
        hidden_size: int,
        image_size: int,
        patch_size: int,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        key, subkey = jax.random.split(key)
        self.patch_embedding = eqx.nn.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="VALID",
            key=key,
            dtype=dtype,
        )
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = eqx.nn.Embedding(
            num_patches, embed_dim, key=subkey, dtype=dtype
        )
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_positions = self.num_patches

    def interpolate_pos_encoding(
        self, embeddings: Array, height: int, width: int
    ) -> Array:
        position_ids = jnp.arange(self.num_positions, dtype=jnp.int32)
        _, num_patches = embeddings.shape
        num_positions, _ = self.position_embedding.weight.shape

        if num_patches == num_positions and height == width:
            return eqx.filter_vmap(self.position_embedding)(position_ids)

        patch_pos_embed = self.position_embedding.weight
        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(self.num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = jnp.transpose(patch_pos_embed, (2, 0, 1))

        patch_pos_embed = jax.image.resize(
            patch_pos_embed,
            shape=(dim, new_height, new_width),
            method="bicubic",
        )

        patch_pos_embed = jnp.transpose(patch_pos_embed, (1, 2, 0)).reshape(1, -1, dim)
        return patch_pos_embed

    def __call__(self, pixel_values: Array, interpolate_pos_encoding=False) -> Array:
        _, img_height, img_width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.astype(dtype=target_dtype))
        width, grid, _ = patch_embeds.shape
        embeddings = patch_embeds.reshape(width, grid**2).T
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, img_height, img_width
            )
        else:
            position_ids = jnp.arange(self.num_positions, dtype=jnp.int32)
            embeddings = embeddings + eqx.filter_vmap(self.position_embedding)(
                position_ids
            )
        return embeddings


class SiglipTextEmbeddings(eqx.Module):
    token_embedding: eqx.nn.Embedding
    position_embedding: eqx.nn.Embedding

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        max_position_embeddings: int,
        key: PRNGKeyArray,
        dtype: Any | None = None,
    ):
        if not dtype:
            dtype = default_floating_dtype()

        assert dtype is not None

        key, token_embd_key, pos_embd_key = jax.random.split(key, 3)
        self.token_embedding = eqx.nn.Embedding(
            vocab_size, embed_dim, dtype=dtype, key=token_embd_key
        )
        self.position_embedding = eqx.nn.Embedding(
            max_position_embeddings, embed_dim, dtype=dtype, key=pos_embd_key
        )

    def __call__(
        self,
        input_ids: Array | None,
        position_ids: Array | None = None,
        inputs_embeds: Array | None = None,
    ) -> Array:
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        else:
            assert inputs_embeds is not None
            seq_length = inputs_embeds.shape[-2]

        max_position_embedding, *_ = self.position_embedding.weight.shape
        if position_ids is None:
            position_ids = jax.lax.dynamic_slice_in_dim(
                jnp.arange(max_position_embedding), start_index=0, slice_size=seq_length
            )

        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = eqx.filter_vmap(eqx.filter_vmap(self.token_embedding))(
                input_ids
            )
            assert inputs_embeds is not None

        position_embeddings = eqx.filter_vmap(self.position_embedding)(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return embeddings


class SiglipAttention(eqx.Module):
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    q_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    embed_dim: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    dropout: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    is_causal: bool = eqx.field(static=True)

    inference: bool

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float,
        dtype: Any | None,
        inference: bool,
        key: PRNGKeyArray,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout
        self.is_causal = False
        self.inference = inference

        key, *subkeys = jax.random.split(key, 5)

        self.k_proj = eqx.nn.Linear(
            self.embed_dim, self.embed_dim, key=subkeys[0], dtype=dtype
        )
        self.v_proj = eqx.nn.Linear(
            self.embed_dim, self.embed_dim, key=subkeys[1], dtype=dtype
        )
        self.q_proj = eqx.nn.Linear(
            self.embed_dim, self.embed_dim, key=subkeys[2], dtype=dtype
        )
        self.out_proj = eqx.nn.Linear(
            self.embed_dim, self.embed_dim, key=subkeys[3], dtype=dtype
        )

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len embed_dim"],
        attention_mask: Array | None,
        output_attentions: bool | None,
        key: PRNGKeyArray | None,
    ) -> tuple[Array, Array | None]:
        seq_length, embed_dim = hidden_states.shape

        attn_output, attn_weights = multi_head_attention_forward(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            use_separate_proj_weight=True,
            embed_dim_to_check=self.embed_dim,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            attn_mask=attention_mask,
            num_heads=self.num_heads,
            inference=self.inference,
            attention_weight_scaling_factor=self.scale,
            dropout_p=0.0 if not self.inference else self.dropout,
            dropout_key=key,
            need_weights=output_attentions if output_attentions is not None else False,
        )

        return attn_output, attn_weights


class SiglipMLP(eqx.Module):
    activation_fn: Callable
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: Callable,
        dtype: Any | None,
        key: PRNGKeyArray,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.activation_fn = hidden_act
        key, subkey = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(hidden_size, intermediate_size, key=key, dtype=dtype)
        self.fc2 = eqx.nn.Linear(
            intermediate_size, hidden_size, key=subkey, dtype=dtype
        )

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
