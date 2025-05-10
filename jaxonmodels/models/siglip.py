import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any
from jaxtyping import Array, PRNGKeyArray

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
            print(f"{embeddings.shape=}, {embeddings[0][:5]=}")
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, img_height, img_width
            )
        else:
            position_ids = jnp.arange(self.num_positions, dtype=jnp.int32)
            embeddings = embeddings + eqx.filter_vmap(self.position_embedding)(
                position_ids
            )
        return embeddings
