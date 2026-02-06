import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from esm.layers.attention import MultiHeadAttention
from esm.layers.blocks import UnifiedTransformerBlock as TorchBlock
from esm.layers.geom_attention import (
    GeometricReasoningOriginalImpl as TorchGeomAttn,
)
from esm.utils.structure.affine3d import (
    Affine3D as TorchAffine3D,
)
from esm.utils.structure.affine3d import (
    RotationMatrix as TorchRotationMatrix,
)
from statedict2pytree.converter import autoconvert

from jaxonmodels.models.esm import (
    Affine3D as JaxAffine3D,
)
from jaxonmodels.models.esm import ESMMultiHeadAttention
from jaxonmodels.models.esm import (
    GeometricReasoningOriginalImpl as JaxGeomAttn,
)
from jaxonmodels.models.esm import (
    RotationMatrix as JaxRotationMatrix,
)
from jaxonmodels.models.esm import UnifiedTransformerBlock as JaxBlock


@pytest.mark.parametrize(
    "d_model, n_heads, seq_len, bias, qk_layernorm",
    [(96, 4, 7, True, True), (960, 15, 7, False, True), (1152, 18, 7, False, True)],
)
def test_mha_implementations(
    d_model: int, n_heads: int, seq_len, bias: bool, qk_layernorm: bool
):
    np.random.seed(42)
    x = np.random.uniform(size=(1, seq_len, d_model))
    x = np.array(x, dtype=np.float32)
    seq_id = np.array([[True for _ in range(seq_len)]])

    torch_mha = MultiHeadAttention(d_model, n_heads, bias, qk_layernorm=qk_layernorm)
    state_dict = torch_mha.state_dict()
    torch_output = torch_mha(torch.from_numpy(x), torch.from_numpy(seq_id))

    key = jax.random.key(42)
    jax_mha = ESMMultiHeadAttention(d_model, n_heads, bias, key=key)
    jax_mha = autoconvert(jax_mha, state_dict)
    x = jnp.array(x, dtype=jnp.float32)
    seq_id = jnp.array([[True for _ in range(seq_len)]])

    jax_output = eqx.filter_vmap(jax_mha)(x, seq_id)

    torch_output = torch_output.detach().numpy()
    jax_output = np.array(jax_output)

    assert np.allclose(torch_output, jax_output, atol=1e-5)


def _make_affines(batch, seq_len, seed=42):
    np.random.seed(seed)
    trans = np.random.randn(batch, seq_len, 3).astype(np.float32)
    rots = np.random.randn(batch, seq_len, 3, 3).astype(np.float32)

    torch_affine = TorchAffine3D(
        torch.from_numpy(trans),
        TorchRotationMatrix(torch.from_numpy(rots)),
    )
    jax_affine = JaxAffine3D(
        jnp.array(trans),
        JaxRotationMatrix(jnp.array(rots)),
    )
    return torch_affine, jax_affine


@pytest.mark.parametrize(
    "c_s, v_heads, seq_len, mask_and_zero, use_seq_id",
    [
        (64, 4, 7, True, False),
        (64, 4, 7, False, True),
        (128, 8, 11, True, True),
        (128, 8, 11, False, False),
    ],
)
def test_geom_attention(c_s, v_heads, seq_len, mask_and_zero, use_seq_id):
    batch = 2
    np.random.seed(42)

    s = np.random.randn(batch, seq_len, c_s).astype(np.float32)
    affine_mask = np.ones((batch, seq_len), dtype=bool)
    affine_mask[:, -1] = False
    chain_id = np.zeros((batch, seq_len), dtype=np.int64)

    if use_seq_id:
        sequence_id = np.zeros((batch, seq_len), dtype=np.int64)
        torch_seq_id = torch.from_numpy(sequence_id)
        jax_seq_id = jnp.array(sequence_id)
    else:
        torch_seq_id = None
        jax_seq_id = None

    torch_affine, jax_affine = _make_affines(batch, seq_len)

    torch_geom = TorchGeomAttn(
        c_s=c_s,
        v_heads=v_heads,
        mask_and_zero_frameless=mask_and_zero,
        bias=False,
    )
    state_dict = torch_geom.state_dict()

    torch_output = (
        torch_geom(
            torch.from_numpy(s),
            torch_affine,
            torch.from_numpy(affine_mask),
            torch_seq_id,
            torch.from_numpy(chain_id),
        )
        .detach()
        .numpy()
    )

    key = jax.random.key(42)
    jax_geom = JaxGeomAttn(
        c_s=c_s,
        v_heads=v_heads,
        mask_and_zero_frameless=mask_and_zero,
        bias=False,
        key=key,
    )
    jax_geom = autoconvert(jax_geom, state_dict)

    def call_single(s_i, affine_i, affine_mask_i, seq_id_i, chain_id_i):
        return jax_geom(s_i, affine_i, affine_mask_i, seq_id_i, chain_id_i)

    jax_s = jnp.array(s)
    jax_affine_mask = jnp.array(affine_mask)
    jax_chain_id = jnp.array(chain_id)

    if jax_seq_id is not None:
        jax_output = np.array(
            eqx.filter_vmap(call_single)(
                jax_s,
                jax_affine,
                jax_affine_mask,
                jax_seq_id,
                jax_chain_id,
            )
        )
    else:
        jax_output = np.array(
            eqx.filter_vmap(call_single, in_axes=(0, 0, 0, None, 0))(
                jax_s,
                jax_affine,
                jax_affine_mask,
                None,
                jax_chain_id,
            )
        )

    assert np.allclose(torch_output, jax_output, atol=1e-4)


def _make_block_inputs(batch, seq_len, d_model, use_frames, use_seq_id, seed=42):
    np.random.seed(seed)
    x = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    chain_id = np.zeros((batch, seq_len), dtype=np.int64)

    if use_seq_id:
        sequence_id = np.zeros((batch, seq_len), dtype=np.int64)
    else:
        sequence_id = None

    if use_frames:
        frames_mask = np.ones((batch, seq_len), dtype=bool)
        frames_mask[:, -1] = False
        trans = np.random.randn(batch, seq_len, 3).astype(np.float32)
        rots = np.random.randn(batch, seq_len, 3, 3).astype(np.float32)
        torch_frames = TorchAffine3D(
            torch.from_numpy(trans), TorchRotationMatrix(torch.from_numpy(rots))
        )
        jax_trans = jnp.array(trans)
        jax_rots = jnp.array(rots)
    else:
        frames_mask = np.ones((batch, seq_len), dtype=bool)
        torch_frames = None
        jax_trans = None
        jax_rots = None

    return x, sequence_id, frames_mask, chain_id, torch_frames, jax_trans, jax_rots


@pytest.mark.parametrize(
    "d_model, n_heads, use_geom, v_heads, bias, expansion_ratio, scaling, qk_ln, ffn_type, use_frames, use_seq_id",
    [
        (64, 4, False, None, False, 8 / 3, 1.0, True, "swiglu", False, True),
        (64, 4, True, 8, False, 8 / 3, 1.1547, True, "swiglu", True, False),
        (64, 4, False, None, True, 4.0, 1.0, False, "gelu", False, True),
        (96, 6, True, 4, False, 8 / 3, 1.1547, True, "swiglu", True, True),
    ],
)
def test_unified_transformer_block(
    d_model,
    n_heads,
    use_geom,
    v_heads,
    bias,
    expansion_ratio,
    scaling,
    qk_ln,
    ffn_type,
    use_frames,
    use_seq_id,
):
    batch = 2
    seq_len = 7

    x, sequence_id, frames_mask, chain_id, torch_frames, jax_trans, jax_rots = (
        _make_block_inputs(batch, seq_len, d_model, use_frames, use_seq_id)
    )

    torch_block = TorchBlock(
        d_model=d_model,
        n_heads=n_heads,
        use_geom_attn=use_geom,
        v_heads=v_heads,
        bias=bias,
        expansion_ratio=expansion_ratio,
        residue_scaling_factor=scaling,
        qk_layernorm=qk_ln,
        ffn_type=ffn_type,
    )
    state_dict = torch_block.state_dict()

    torch_output = (
        torch_block(
            torch.from_numpy(x),
            torch.from_numpy(sequence_id) if sequence_id is not None else None,
            torch_frames,
            torch.from_numpy(frames_mask),
            torch.from_numpy(chain_id),
        )
        .detach()
        .numpy()
    )

    key = jax.random.key(42)
    jax_block = JaxBlock(
        d_model=d_model,
        n_heads=n_heads,
        use_geom_attn=use_geom,
        v_heads=v_heads,
        bias=bias,
        expansion_ratio=expansion_ratio,
        residue_scaling_factor=scaling,
        qk_layernorm=qk_ln,
        ffn_type=ffn_type,
        key=key,
    )
    jax_block = autoconvert(jax_block, state_dict)

    jax_x = jnp.array(x)
    jax_frames_mask = jnp.array(frames_mask)
    jax_chain_id = jnp.array(chain_id)
    jax_seq_id = jnp.array(sequence_id) if sequence_id is not None else None

    def call_single(x_i, trans_i, rots_i, frames_mask_i, seq_id_i, chain_id_i):
        if trans_i is not None:
            frames_i = JaxAffine3D(trans_i, JaxRotationMatrix(rots_i))
        else:
            frames_i = None
        return jax_block(x_i, seq_id_i, frames_i, frames_mask_i, chain_id_i)

    if use_frames:
        if jax_seq_id is not None:
            jax_output = np.array(
                eqx.filter_vmap(call_single)(
                    jax_x,
                    jax_trans,
                    jax_rots,
                    jax_frames_mask,
                    jax_seq_id,
                    jax_chain_id,
                )
            )
        else:
            jax_output = np.array(
                eqx.filter_vmap(call_single, in_axes=(0, 0, 0, 0, None, 0))(
                    jax_x,
                    jax_trans,
                    jax_rots,
                    jax_frames_mask,
                    None,
                    jax_chain_id,
                )
            )
    else:
        if jax_seq_id is not None:
            jax_output = np.array(
                eqx.filter_vmap(call_single, in_axes=(0, None, None, 0, 0, 0))(
                    jax_x,
                    None,
                    None,
                    jax_frames_mask,
                    jax_seq_id,
                    jax_chain_id,
                )
            )
        else:
            jax_output = np.array(
                eqx.filter_vmap(call_single, in_axes=(0, None, None, 0, None, 0))(
                    jax_x,
                    None,
                    None,
                    jax_frames_mask,
                    None,
                    jax_chain_id,
                )
            )

    assert np.allclose(torch_output, jax_output, atol=1e-3)
