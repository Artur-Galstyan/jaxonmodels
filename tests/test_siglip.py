import equinox as eqx
import jax
import numpy as np
import torch
from pydantic import BaseModel
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionEmbeddings as TorchSiglipVisionEmbeddings,
)

from jaxonmodels.models.siglip import SiglipVisionEmbeddings
from jaxonmodels.statedict2pytree import s2p


def test_SiglipVisionEmbeddings():
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
    assert np.allclose(t_out.detach().numpy(), np.array(j_out), atol=1e-4)

    # test_array = np.ones(shape=(1, 3, 384, 384))
    # t_out = torch_embs(torch.from_numpy(test_array), True)

    # jax_embs = functools.partial(jax_embs, interpolate_pos_encoding=True)
    # j_out = eqx.filter_vmap(jax_embs)(test_array)

    # assert np.allclose(t_out.detach().numpy(), np.array(j_out), atol=1e-4)
