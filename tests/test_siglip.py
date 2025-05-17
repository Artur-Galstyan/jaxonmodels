import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from pydantic import BaseModel
from transformers.models.siglip.modeling_siglip import (
    SiglipAttention as TorchSiglipAttention,
)
from transformers.models.siglip.modeling_siglip import (
    SiglipTextEmbeddings as TorchSiglipTextEmbeddings,
)
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionEmbeddings as TorchSiglipVisionEmbeddings,
)

from jaxonmodels.models.siglip import (
    SiglipAttention,
    SiglipTextEmbeddings,
    SiglipVisionEmbeddings,
)
from jaxonmodels.statedict2pytree import s2p


def test_SiglipVisionEmbeddings():
    np.random.seed(42)
    torch.manual_seed(42)

    class TempConfig(BaseModel):
        hidden_size: int
        image_size: int
        num_channels: int
        patch_size: int

    config = TempConfig(hidden_size=768, image_size=224, num_channels=3, patch_size=16)

    torch_embs = TorchSiglipVisionEmbeddings(config)  # pyright: ignore

    jax_embs = SiglipVisionEmbeddings(
        num_channels=config.num_channels,
        embed_dim=config.hidden_size,
        hidden_size=config.hidden_size,
        image_size=config.image_size,
        patch_size=config.patch_size,
        key=jax.random.key(44),
    )

    jax_embs = s2p.autoconvert(jax_embs, torch_embs.state_dict())

    test_array = np.ones(shape=(1, 3, 224, 224))
    t_out = torch_embs(torch.from_numpy(test_array))
    j_out = eqx.filter_vmap(jax_embs)(test_array)
    assert t_out.shape == j_out.shape

    assert np.allclose(t_out.detach().numpy(), np.array(j_out), atol=1e-5)

    # the following test would fail because
    # pytorch's interpolate function and jax's image.resize function
    # use different kernels for the bicubic interpolation
    # test_array = np.ones(shape=(1, 3, 384, 384))
    # t_out = torch_embs(torch.from_numpy(test_array), True)

    # jax_embs = functools.partial(jax_embs, interpolate_pos_encoding=True)
    # j_out = eqx.filter_vmap(jax_embs)(test_array)

    # print(j_out[0][:5])
    # print(t_out[0][:5])

    # print(np.allclose(t_out.detach().numpy(), np.array(j_out), atol=1e-3))


def test_SiglipTextEmbeddings():
    np.random.seed(42)
    torch.manual_seed(42)

    class TempConfig(BaseModel):
        attention_dropout: float | None = None
        bos_token_id: int | None = None
        eos_token_id: int | None = None
        hidden_act: str | None = None
        hidden_size: int | None = None
        intermediate_size: int | None = None
        layer_norm_eps: float | None = None
        max_position_embeddings: int | None = None
        model_type: str | None = None
        num_attention_heads: int | None = None
        num_hidden_layers: int | None = None
        pad_token_id: int | None = None
        projection_size: int | None = None
        torch_dtype: str | None = None
        vocab_size: int | None = None
        image_size: int | None = None
        num_channels: int | None = None
        patch_size: int | None = None

    config = TempConfig(hidden_size=768, vocab_size=32000, max_position_embeddings=64)

    torch_embeds = TorchSiglipTextEmbeddings(config)  # pyright: ignore

    jax_embeds = SiglipTextEmbeddings(
        embed_dim=config.hidden_size,  # pyright: ignore
        vocab_size=config.vocab_size,  # pyright: ignore
        max_position_embeddings=config.max_position_embeddings,  # pyright: ignore
        key=jax.random.key(44),
    )  # pyright: ignore

    jax_embeds = s2p.autoconvert(jax_embeds, torch_embeds.state_dict())
    assert config.max_position_embeddings is not None
    input_ids = np.random.randint(
        low=0, high=1000, size=(3, config.max_position_embeddings)
    )

    t_out = torch_embeds(torch.from_numpy(input_ids))

    j_out = jax_embeds(jnp.array(input_ids))

    assert np.allclose(t_out.detach().numpy(), np.array(j_out), atol=1e-5)


def test_attention_shapes():
    np.random.seed(42)
    torch.manual_seed(42)

    class AttnTempConfig:
        attention_dropout: float | None = None
        bos_token_id: int | None = None
        eos_token_id: int | None = None
        hidden_act: str | None = None
        hidden_size: int | None = None
        intermediate_size: int | None = None
        layer_norm_eps: float | None = None
        max_position_embeddings: int | None = None
        model_type: str | None = None
        num_attention_heads: int | None = None
        num_hidden_layers: int | None = None
        pad_token_id: int | None = None
        projection_size: int | None = None
        torch_dtype: str | None = None
        vocab_size: int | None = None
        image_size: int | None = None
        num_channels: int | None = None
        patch_size: int | None = None
        _attn_implementation: str | None = None

        def __init__(
            self,
            attention_dropout=None,
            bos_token_id=None,
            eos_token_id=None,
            hidden_act=None,
            hidden_size=None,
            intermediate_size=None,
            layer_norm_eps=None,
            max_position_embeddings=None,
            model_type=None,
            num_attention_heads=None,
            num_hidden_layers=None,
            pad_token_id=None,
            projection_size=None,
            torch_dtype=None,
            vocab_size=None,
            image_size=None,
            num_channels=None,
            patch_size=None,
            _attn_implementation=None,
        ):
            self.attention_dropout = attention_dropout
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            self.hidden_act = hidden_act
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.layer_norm_eps = layer_norm_eps
            self.max_position_embeddings = max_position_embeddings
            self.model_type = model_type
            self.num_attention_heads = num_attention_heads
            self.num_hidden_layers = num_hidden_layers
            self.pad_token_id = pad_token_id
            self.projection_size = projection_size
            self.torch_dtype = torch_dtype
            self.vocab_size = vocab_size
            self.image_size = image_size
            self.num_channels = num_channels
            self.patch_size = patch_size
            self._attn_implementation = _attn_implementation

    config = AttnTempConfig(
        hidden_size=768,
        vocab_size=32000,
        max_position_embeddings=64,
        attention_dropout=0.0,
        num_attention_heads=12,
        _attn_implementation="eager",
    )
    print(config._attn_implementation)
    torch_attention = TorchSiglipAttention(config)  # pyright: ignore

    jax_attention = SiglipAttention(
        embed_dim=config.hidden_size,  # pyright: ignore
        num_heads=config.num_attention_heads,  # pyright: ignore
        attention_dropout=config.attention_dropout,  # pyright: ignore
        dtype=None,
        inference=False,
        key=jax.random.key(1),
    )
    hidden_states = jnp.ones(shape=(4, 128, config.hidden_size))
    hidden_states = np.array(hidden_states)
    output_attentions = False
    attn_mask = None

    t_attn_output, t_attn_weights = torch_attention(
        torch.from_numpy(hidden_states), attn_mask, output_attentions
    )

    print(f"{t_attn_output.shape=}")
    if t_attn_weights:
        print(f"{t_attn_weights.shape=}")

    jax_attention_pt = functools.partial(
        jax_attention,
        attention_mask=None,
        output_attentions=output_attentions,
        key=jax.random.key(44),
    )
    j_attn_output, j_attn_weights = eqx.filter_vmap(jax_attention_pt)(hidden_states)
