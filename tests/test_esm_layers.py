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
from esm.layers.transformer_stack import TransformerStack as TorchTransformerStack
from esm.models.esm3 import ESM3 as TorchESM3
from esm.models.esm3 import EncodeInputs as TorchEncodeInputs
from esm.models.esm3 import OutputHeads as TorchOutputHeads
from esm.models.esmc import ESMC as TorchESMC
from esm.tokenization import EsmSequenceTokenizer
from esm.utils.structure.affine3d import (
    Affine3D as TorchAffine3D,
)
from esm.utils.structure.affine3d import (
    RotationMatrix as TorchRotationMatrix,
)
from statedict2pytree.converter import autoconvert

from jaxonmodels.models.esm import ESM3 as JaxESM3
from jaxonmodels.models.esm import ESMC as JaxESMC
from jaxonmodels.models.esm import (
    Affine3D as JaxAffine3D,
)
from jaxonmodels.models.esm import EncodeInputs as JaxEncodeInputs
from jaxonmodels.models.esm import ESMMultiHeadAttention
from jaxonmodels.models.esm import (
    GeometricReasoningOriginalImpl as JaxGeomAttn,
)
from jaxonmodels.models.esm import OutputHeads as JaxOutputHeads
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


def _make_stack_inputs(batch, seq_len, d_model, use_frames, use_seq_id, seed=42):
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


from jaxonmodels.models.esm import TransformerStack as JaxTransformerStack


@pytest.mark.parametrize(
    "d_model, n_heads, v_heads, n_layers, n_layers_geom, use_frames, use_seq_id",
    [
        (64, 4, None, 3, 0, False, True),
        (64, 4, 4, 3, 1, True, False),
    ],
)
def test_transformer_stack(
    d_model,
    n_heads,
    v_heads,
    n_layers,
    n_layers_geom,
    use_frames,
    use_seq_id,
):
    batch = 2
    seq_len = 7

    x, sequence_id, frames_mask, chain_id, torch_frames, jax_trans, jax_rots = (
        _make_stack_inputs(batch, seq_len, d_model, use_frames, use_seq_id)
    )

    torch_stack = TorchTransformerStack(
        d_model=d_model,
        n_heads=n_heads,
        v_heads=v_heads,
        n_layers=n_layers,
        n_layers_geom=n_layers_geom,
    )
    state_dict = torch_stack.state_dict()

    torch_x, torch_embed, torch_hiddens = torch_stack(
        torch.from_numpy(x),
        torch.from_numpy(sequence_id) if sequence_id is not None else None,
        torch_frames,
        torch.from_numpy(frames_mask),
        torch.from_numpy(chain_id),
    )
    torch_x = torch_x.detach().numpy()
    torch_embed = torch_embed.detach().numpy()

    key = jax.random.key(42)
    jax_stack = JaxTransformerStack(
        d_model=d_model,
        n_heads=n_heads,
        v_heads=v_heads,
        n_layers=n_layers,
        n_layers_geom=n_layers_geom,
        key=key,
    )
    jax_stack = autoconvert(jax_stack, state_dict)

    jax_x_in = jnp.array(x)
    jax_frames_mask = jnp.array(frames_mask)
    jax_chain_id = jnp.array(chain_id)
    jax_seq_id = jnp.array(sequence_id) if sequence_id is not None else None

    def call_single(x_i, trans_i, rots_i, frames_mask_i, seq_id_i, chain_id_i):
        if trans_i is not None:
            frames_i = JaxAffine3D(trans_i, JaxRotationMatrix(rots_i))
        else:
            frames_i = None
        return jax_stack(x_i, seq_id_i, frames_i, frames_mask_i, chain_id_i)

    if use_frames:
        if jax_seq_id is not None:
            jax_out = eqx.filter_vmap(call_single)(
                jax_x_in,
                jax_trans,
                jax_rots,
                jax_frames_mask,
                jax_seq_id,
                jax_chain_id,
            )
        else:
            jax_out = eqx.filter_vmap(call_single, in_axes=(0, 0, 0, 0, None, 0))(
                jax_x_in,
                jax_trans,
                jax_rots,
                jax_frames_mask,
                None,
                jax_chain_id,
            )
    else:
        if jax_seq_id is not None:
            jax_out = eqx.filter_vmap(call_single, in_axes=(0, None, None, 0, 0, 0))(
                jax_x_in,
                None,
                None,
                jax_frames_mask,
                jax_seq_id,
                jax_chain_id,
            )
        else:
            jax_out = eqx.filter_vmap(call_single, in_axes=(0, None, None, 0, None, 0))(
                jax_x_in,
                None,
                None,
                jax_frames_mask,
                None,
                jax_chain_id,
            )

    jax_x_out, jax_embed, jax_hiddens = jax_out

    assert np.allclose(torch_x, np.array(jax_x_out), atol=1e-3)
    assert np.allclose(torch_embed, np.array(jax_embed), atol=1e-3)
    assert len(torch_hiddens) == len(jax_hiddens)
    for i in range(len(torch_hiddens)):
        assert np.allclose(
            torch_hiddens[i].detach().numpy(), np.array(jax_hiddens[i]), atol=1e-3
        )


@pytest.mark.parametrize(
    "d_model, n_heads, n_layers, seq_len",
    [
        (64, 4, 2, 7),
        (96, 6, 3, 11),
    ],
)
def test_esmc(d_model, n_heads, n_layers, seq_len):
    batch = 2
    np.random.seed(42)

    tokenizer = EsmSequenceTokenizer()
    tokens = np.random.randint(0, 33, size=(batch, seq_len)).astype(np.int64)
    tokens[:, 0] = 0
    tokens[:, -1] = 2

    torch_model = TorchESMC(d_model, n_heads, n_layers, tokenizer)
    state_dict = torch_model.state_dict()

    with torch.no_grad():
        output = torch_model(torch.from_numpy(tokens))
        torch_logits, torch_embed, torch_hiddens = (
            output.sequence_logits,
            output.embeddings,
            output.hidden_states,
        )
    torch_logits = torch_logits.numpy()
    torch_embed = torch_embed.numpy()
    torch_hiddens = torch_hiddens.numpy()

    key = jax.random.key(42)
    jax_model = JaxESMC(d_model, n_heads, n_layers, key=key)
    jax_model = autoconvert(jax_model, state_dict)

    jax_tokens = jnp.array(tokens)

    def call_single(tokens_i):
        return jax_model(tokens_i)

    jax_logits, jax_embed, jax_hiddens = eqx.filter_vmap(call_single)(jax_tokens)
    jax_hiddens = np.array(jax_hiddens).transpose(1, 0, 2, 3)
    assert np.allclose(torch_hiddens, jax_hiddens, atol=1e-3)

    assert np.allclose(torch_logits, np.array(jax_logits), atol=1e-3)
    assert np.allclose(torch_embed, np.array(jax_embed), atol=1e-3)


@pytest.mark.parametrize(
    "d_model, seq_len",
    [
        (64, 7),
        (128, 11),
    ],
)
def test_encode_inputs(d_model, seq_len):
    batch = 2
    np.random.seed(42)

    sequence_tokens = np.random.randint(0, 64, size=(batch, seq_len)).astype(np.int64)
    structure_tokens = np.random.randint(0, 4096, size=(batch, seq_len)).astype(
        np.int64
    )
    ss8_tokens = np.random.randint(0, 11, size=(batch, seq_len)).astype(np.int64)
    sasa_tokens = np.random.randint(0, 19, size=(batch, seq_len)).astype(np.int64)
    function_tokens = np.random.randint(0, 260, size=(batch, seq_len, 8)).astype(
        np.int64
    )
    residue_annotation_tokens = np.random.randint(
        0, 1478, size=(batch, seq_len, 16)
    ).astype(np.int64)
    average_plddt = np.float32(1.0)
    per_res_plddt = np.random.rand(batch, seq_len).astype(np.float32)

    torch_encoder = TorchEncodeInputs(d_model)
    state_dict = torch_encoder.state_dict()

    torch_output = (
        torch_encoder(
            torch.from_numpy(sequence_tokens),
            torch.from_numpy(structure_tokens),
            torch.tensor(average_plddt),
            torch.from_numpy(per_res_plddt),
            torch.from_numpy(ss8_tokens),
            torch.from_numpy(sasa_tokens),
            torch.from_numpy(function_tokens),
            torch.from_numpy(residue_annotation_tokens),
        )
        .detach()
        .numpy()
    )

    key = jax.random.key(42)
    jax_encoder = JaxEncodeInputs(d_model, key=key)
    jax_encoder = autoconvert(jax_encoder, state_dict)

    def call_single(seq, struct, res_plddt, ss8, sasa, func, residue):
        return jax_encoder(
            seq, struct, average_plddt, res_plddt, ss8, sasa, func, residue
        )

    jax_output = np.array(
        eqx.filter_vmap(call_single)(
            jnp.array(sequence_tokens),
            jnp.array(structure_tokens),
            jnp.array(per_res_plddt),
            jnp.array(ss8_tokens),
            jnp.array(sasa_tokens),
            jnp.array(function_tokens),
            jnp.array(residue_annotation_tokens),
        )
    )

    assert np.allclose(torch_output, jax_output, atol=1e-4)


@pytest.mark.parametrize(
    "d_model, seq_len",
    [
        (64, 7),
        (128, 11),
    ],
)
def test_output_heads(d_model, seq_len):
    batch = 2
    np.random.seed(42)

    x = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    embed = np.random.randn(batch, seq_len, d_model).astype(np.float32)

    torch_heads = TorchOutputHeads(d_model)
    state_dict = torch_heads.state_dict()

    torch_out = torch_heads(torch.from_numpy(x), torch.from_numpy(embed))
    torch_results = [
        torch_out.sequence_logits.detach().numpy(),
        torch_out.structure_logits.detach().numpy(),
        torch_out.secondary_structure_logits.detach().numpy(),
        torch_out.sasa_logits.detach().numpy(),
        torch_out.function_logits.detach().numpy(),
        torch_out.residue_logits.detach().numpy(),
        torch_out.embeddings.detach().numpy(),
    ]

    key = jax.random.key(42)
    jax_heads = JaxOutputHeads(d_model, key=key)
    jax_heads = autoconvert(jax_heads, state_dict)

    def call_single(x_i, embed_i):
        return jax_heads(x_i, embed_i)

    jax_out = eqx.filter_vmap(call_single)(jnp.array(x), jnp.array(embed))

    for i in range(len(torch_results)):
        assert np.allclose(torch_results[i], np.array(jax_out[i]), atol=1e-3), (
            f"Mismatch at output index {i}"
        )


class _MockSeqTok:
    mask_token_id = 32


class _MockTokenizers:
    sequence = _MockSeqTok()


@pytest.mark.parametrize(
    "d_model, n_heads, v_heads, n_layers, seq_len",
    [
        (64, 4, 4, 2, 7),
        (96, 6, 4, 3, 11),
    ],
)
def test_esm3_forward(d_model, n_heads, v_heads, n_layers, seq_len):
    batch = 2
    np.random.seed(42)

    sequence_tokens = np.random.randint(3, 31, size=(batch, seq_len)).astype(np.int64)
    sequence_tokens[:, 0] = 0
    sequence_tokens[:, -1] = 2
    structure_tokens = np.random.randint(0, 4096, size=(batch, seq_len)).astype(
        np.int64
    )
    ss8_tokens = np.full((batch, seq_len), 8, dtype=np.int64)
    sasa_tokens = np.full((batch, seq_len), 16, dtype=np.int64)
    function_tokens = np.full((batch, seq_len, 8), 0, dtype=np.int64)
    residue_annotation_tokens = np.full((batch, seq_len, 16), 0, dtype=np.int64)
    average_plddt = np.float32(1.0)
    per_res_plddt = np.zeros((batch, seq_len), dtype=np.float32)
    structure_coords = np.full((batch, seq_len, 3, 3), np.nan, dtype=np.float32)
    chain_id = np.zeros((batch, seq_len), dtype=np.int64)

    torch_model = TorchESM3(
        d_model,
        n_heads,
        v_heads,
        n_layers,
        structure_encoder_fn=None,  # ty: ignore[invalid-argument-type]
        structure_decoder_fn=None,  # ty: ignore[invalid-argument-type]
        function_decoder_fn=None,  # ty: ignore[invalid-argument-type]
        tokenizers=_MockTokenizers(),  # ty: ignore[invalid-argument-type]
    )
    state_dict = torch_model.state_dict()

    with torch.no_grad():
        torch_out = torch_model(
            sequence_tokens=torch.from_numpy(sequence_tokens),
            structure_tokens=torch.from_numpy(structure_tokens),
            ss8_tokens=torch.from_numpy(ss8_tokens),
            sasa_tokens=torch.from_numpy(sasa_tokens),
            function_tokens=torch.from_numpy(function_tokens),
            residue_annotation_tokens=torch.from_numpy(residue_annotation_tokens),
            average_plddt=torch.tensor(average_plddt),
            per_res_plddt=torch.from_numpy(per_res_plddt),
            structure_coords=torch.from_numpy(structure_coords),
            chain_id=torch.from_numpy(chain_id),
        )

    torch_seq_logits = torch_out.sequence_logits.numpy()
    torch_struct_logits = torch_out.structure_logits.numpy()
    torch_embed = torch_out.embeddings.numpy()

    key = jax.random.key(42)
    jax_model = JaxESM3(d_model, n_heads, v_heads, n_layers, key=key)
    jax_model = autoconvert(jax_model, state_dict)

    def call_single(seq, struct, ss8, sasa, func, residue, plddt, coords, cid):
        return jax_model(
            sequence_tokens=seq,
            structure_tokens=struct,
            ss8_tokens=ss8,
            sasa_tokens=sasa,
            function_tokens=func,
            residue_annotation_tokens=residue,
            average_plddt=average_plddt,
            per_res_plddt=plddt,
            structure_coords=coords,
            chain_id=cid,
        )

    jax_out = eqx.filter_vmap(call_single)(
        jnp.array(sequence_tokens),
        jnp.array(structure_tokens),
        jnp.array(ss8_tokens),
        jnp.array(sasa_tokens),
        jnp.array(function_tokens),
        jnp.array(residue_annotation_tokens),
        jnp.array(per_res_plddt),
        jnp.array(structure_coords),
        jnp.array(chain_id),
    )

    jax_seq_logits = np.array(jax_out[0])
    jax_struct_logits = np.array(jax_out[1])
    jax_embed = np.array(jax_out[6])

    assert np.allclose(torch_seq_logits, jax_seq_logits, atol=1e-3)
    assert np.allclose(torch_struct_logits, jax_struct_logits, atol=1e-3)
    assert np.allclose(torch_embed, jax_embed, atol=1e-3)


def test_esm3_forward_debug():
    d_model, n_heads, v_heads, n_layers, seq_len = 64, 4, 4, 2, 7
    batch = 2
    np.random.seed(42)

    sequence_tokens = np.random.randint(3, 31, size=(batch, seq_len)).astype(np.int64)
    sequence_tokens[:, 0] = 0
    sequence_tokens[:, -1] = 2
    structure_tokens = np.random.randint(0, 4096, size=(batch, seq_len)).astype(
        np.int64
    )
    ss8_tokens = np.full((batch, seq_len), 8, dtype=np.int64)
    sasa_tokens = np.full((batch, seq_len), 16, dtype=np.int64)
    function_tokens = np.full((batch, seq_len, 8), 0, dtype=np.int64)
    residue_annotation_tokens = np.full((batch, seq_len, 16), 0, dtype=np.int64)
    average_plddt = np.float32(1.0)
    per_res_plddt = np.zeros((batch, seq_len), dtype=np.float32)
    structure_coords = np.full((batch, seq_len, 3, 3), np.nan, dtype=np.float32)
    chain_id = np.zeros((batch, seq_len), dtype=np.int64)

    torch_model = TorchESM3(
        d_model,
        n_heads,
        v_heads,
        n_layers,
        structure_encoder_fn=None,  # ty: ignore[invalid-argument-type]
        structure_decoder_fn=None,  # ty: ignore[invalid-argument-type]
        function_decoder_fn=None,  # ty: ignore[invalid-argument-type]
        tokenizers=_MockTokenizers(),  # ty: ignore[invalid-argument-type]
    )
    state_dict = torch_model.state_dict()

    key = jax.random.key(42)
    jax_model = JaxESM3(d_model, n_heads, v_heads, n_layers, key=key)
    jax_model = autoconvert(jax_model, state_dict)

    sq_t = torch.from_numpy(sequence_tokens)
    st_t = torch.from_numpy(structure_tokens).clone()
    st_t = st_t.masked_fill(st_t == -1, 4096)
    st_t = st_t.masked_fill(sq_t == 0, 4098)
    st_t = st_t.masked_fill(sq_t == 1, 4097)
    st_t = st_t.masked_fill(sq_t == 2, 4099)
    st_t = st_t.masked_fill(sq_t == 31, 4100)

    sq_j = jnp.array(sequence_tokens)
    st_j = jnp.array(structure_tokens)
    st_j = jnp.where(st_j == -1, 4096, st_j)
    st_j = jnp.where(sq_j == 0, 4098, st_j)
    st_j = jnp.where(sq_j == 1, 4097, st_j)
    st_j = jnp.where(sq_j == 2, 4099, st_j)
    st_j = jnp.where(sq_j == 31, 4100, st_j)

    print("=== Step 1: Modified structure tokens ===")
    print("Match:", np.array_equal(st_t.numpy(), np.array(st_j)))

    with torch.no_grad():
        torch_enc = (
            torch_model.encoder(
                sq_t,
                st_t,
                torch.tensor(average_plddt),
                torch.from_numpy(per_res_plddt),
                torch.from_numpy(ss8_tokens),
                torch.from_numpy(sasa_tokens),
                torch.from_numpy(function_tokens),
                torch.from_numpy(residue_annotation_tokens),
            )
            .detach()
            .numpy()
        )

    def jax_encode_single(seq, struct, plddt, ss8, sasa, func, residue):
        return jax_model.encoder(
            seq, struct, average_plddt, plddt, ss8, sasa, func, residue
        )

    jax_enc = np.array(
        eqx.filter_vmap(jax_encode_single)(
            sq_j,
            st_j,
            jnp.array(per_res_plddt),
            jnp.array(ss8_tokens),
            jnp.array(sasa_tokens),
            jnp.array(function_tokens),
            jnp.array(residue_annotation_tokens),
        )
    )

    print("=== Step 2: Encoder output ===")
    print("Match:", np.allclose(torch_enc, jax_enc, atol=1e-4))
    print("Max diff:", np.abs(torch_enc - jax_enc).max())

    from esm.utils.structure.affine3d import (
        build_affine3d_from_coordinates as torch_build_affine,
    )

    from jaxonmodels.models.esm import (
        build_affine3d_from_coordinates as jax_build_affine,
    )

    torch_coords = torch.from_numpy(structure_coords)[..., :3, :]
    torch_affine, torch_affine_mask = torch_build_affine(torch_coords)
    torch_affine_tensor = torch_affine.tensor.detach().numpy()
    torch_affine_mask_np = torch_affine_mask.numpy()

    def jax_build_single(c):
        affine, mask = jax_build_affine(c)
        return affine.tensor, mask

    jax_affine_tensor, jax_affine_mask = eqx.filter_vmap(jax_build_single)(
        jnp.array(structure_coords)[..., :3, :]
    )

    print("=== Step 3: Affine ===")
    print(
        "Mask match:", np.array_equal(torch_affine_mask_np, np.array(jax_affine_mask))
    )
    print(
        "Tensor match:",
        np.allclose(
            torch_affine_tensor, np.array(jax_affine_tensor), atol=1e-5, equal_nan=True
        ),
    )
    print(
        "Affine max diff:",
        np.nanmax(np.abs(torch_affine_tensor - np.array(jax_affine_tensor))),
    )

    with torch.no_grad():
        torch_x = torch.from_numpy(torch_enc)
        torch_trans_out, torch_embed, torch_hiddens = torch_model.transformer(
            torch_x,
            None,
            torch_affine,
            torch_affine_mask,
            torch.from_numpy(chain_id),
        )

    from jaxonmodels.models.esm import Affine3D as JaxAffine3D
    from jaxonmodels.models.esm import RotationMatrix as JaxRotationMatrix

    def jax_transformer_single(x_i, affine_t_i, affine_m_i, chain_id_i):
        affine_i = JaxAffine3D.from_tensor(affine_t_i)
        return jax_model.transformer(x_i, None, affine_i, affine_m_i, chain_id_i)

    jax_trans_out, jax_embed, jax_hiddens = eqx.filter_vmap(
        jax_transformer_single, in_axes=(0, 0, 0, 0)
    )(
        jnp.array(torch_enc),
        jax_affine_tensor,
        jax_affine_mask,
        jnp.array(chain_id),
    )

    print("=== Step 4: Transformer output ===")
    print(
        "Match:",
        np.allclose(
            torch_trans_out.detach().numpy(), np.array(jax_trans_out), atol=1e-3
        ),
    )
    print(
        "Max diff:",
        np.abs(torch_trans_out.detach().numpy() - np.array(jax_trans_out)).max(),
    )

    for i, (th, jh) in enumerate(zip(torch_hiddens, jax_hiddens)):
        diff = np.abs(th.detach().numpy() - np.array(jh)).max()
        print(f"Hidden layer {i} max diff: {diff}")


def test_esm3_forward_no_vmap():
    d_model, n_heads, v_heads, n_layers, seq_len = 64, 4, 4, 2, 7
    np.random.seed(42)

    sequence_tokens = np.random.randint(3, 31, size=(1, seq_len)).astype(np.int64)
    sequence_tokens[:, 0] = 0
    sequence_tokens[:, -1] = 2
    structure_tokens = np.random.randint(0, 4096, size=(1, seq_len)).astype(np.int64)
    ss8_tokens = np.full((1, seq_len), 8, dtype=np.int64)
    sasa_tokens = np.full((1, seq_len), 16, dtype=np.int64)
    function_tokens = np.full((1, seq_len, 8), 0, dtype=np.int64)
    residue_annotation_tokens = np.full((1, seq_len, 16), 0, dtype=np.int64)
    average_plddt = np.float32(1.0)
    per_res_plddt = np.zeros((1, seq_len), dtype=np.float32)
    structure_coords = np.full((1, seq_len, 3, 3), np.nan, dtype=np.float32)
    chain_id = np.zeros((1, seq_len), dtype=np.int64)

    torch_model = TorchESM3(
        d_model,
        n_heads,
        v_heads,
        n_layers,
        structure_encoder_fn=None,  # ty: ignore[invalid-argument-type]
        structure_decoder_fn=None,  # ty: ignore[invalid-argument-type]
        function_decoder_fn=None,  # ty: ignore[invalid-argument-type]
        tokenizers=_MockTokenizers(),  # ty: ignore[invalid-argument-type]
    )
    state_dict = torch_model.state_dict()

    with torch.no_grad():
        torch_out = torch_model(
            sequence_tokens=torch.from_numpy(sequence_tokens),
            structure_tokens=torch.from_numpy(structure_tokens),
            ss8_tokens=torch.from_numpy(ss8_tokens),
            sasa_tokens=torch.from_numpy(sasa_tokens),
            function_tokens=torch.from_numpy(function_tokens),
            residue_annotation_tokens=torch.from_numpy(residue_annotation_tokens),
            average_plddt=torch.tensor(average_plddt),
            per_res_plddt=torch.from_numpy(per_res_plddt),
            structure_coords=torch.from_numpy(structure_coords),
            chain_id=torch.from_numpy(chain_id),
        )
    torch_seq_logits = torch_out.sequence_logits.numpy()

    key = jax.random.key(42)
    jax_model = JaxESM3(d_model, n_heads, v_heads, n_layers, key=key)
    jax_model = autoconvert(jax_model, state_dict)

    jax_out = jax_model(
        sequence_tokens=jnp.array(sequence_tokens[0]),
        structure_tokens=jnp.array(structure_tokens[0]),
        ss8_tokens=jnp.array(ss8_tokens[0]),
        sasa_tokens=jnp.array(sasa_tokens[0]),
        function_tokens=jnp.array(function_tokens[0]),
        residue_annotation_tokens=jnp.array(residue_annotation_tokens[0]),
        average_plddt=average_plddt,
        per_res_plddt=jnp.array(per_res_plddt[0]),
        structure_coords=jnp.array(structure_coords[0]),
        chain_id=jnp.array(chain_id[0]),
    )

    jax_seq_logits = np.array(jax_out[0])

    print("=== No-vmap test ===")
    print("Match:", np.allclose(torch_seq_logits[0], jax_seq_logits, atol=1e-3))
    print("Max diff:", np.abs(torch_seq_logits[0] - jax_seq_logits).max())
    print("Has NaN (torch):", np.any(np.isnan(torch_seq_logits)))
    print("Has NaN (jax):", np.any(np.isnan(jax_seq_logits)))

    assert np.allclose(torch_seq_logits[0], jax_seq_logits, atol=1e-3)
