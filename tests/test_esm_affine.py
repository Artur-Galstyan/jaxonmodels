import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from esm.utils.structure.affine3d import (
    Affine3D as TorchAffine3D,
)
from esm.utils.structure.affine3d import (
    RotationMatrix as TorchRotationMatrix,
)
from esm.utils.structure.affine3d import (
    RotationQuat as TorchRotationQuat,
)
from esm.utils.structure.affine3d import (
    _graham_schmidt as torch_graham_schmidt,
)
from esm.utils.structure.affine3d import (
    _quat_invert as torch_quat_invert,
)
from esm.utils.structure.affine3d import (
    _quat_mult as torch_quat_mult,
)
from esm.utils.structure.affine3d import (
    _quat_rotation as torch_quat_rotation,
)
from esm.utils.structure.affine3d import (
    _sqrt_subgradient as torch_sqrt_subgradient,
)
from esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates as torch_build_affine,
)

from jaxonmodels.models.esm import (
    Affine3D as JaxAffine3D,
)
from jaxonmodels.models.esm import (
    RotationMatrix as JaxRotationMatrix,
)
from jaxonmodels.models.esm import (
    RotationQuat as JaxRotationQuat,
)
from jaxonmodels.models.esm import (
    _graham_schmidt as jax_graham_schmidt,
)
from jaxonmodels.models.esm import (
    _quat_invert as jax_quat_invert,
)
from jaxonmodels.models.esm import (
    _quat_mult as jax_quat_mult,
)
from jaxonmodels.models.esm import (
    _quat_rotation as jax_quat_rotation,
)
from jaxonmodels.models.esm import (
    _sqrt_subgradient as jax_sqrt_subgradient,
)
from jaxonmodels.models.esm import build_affine3d_from_coordinates as jax_build_affine


def _rand(*shape, seed=42):
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32)


def test_quat_mult():
    a = _rand(5, 4)
    b = _rand(5, 4, seed=43)
    torch_out = torch_quat_mult(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    jax_out = np.array(jax_quat_mult(jnp.array(a), jnp.array(b)))
    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_quat_mult_batched():
    a = _rand(3, 5, 4)
    b = _rand(3, 5, 4, seed=43)
    torch_out = torch_quat_mult(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    jax_out = np.array(jax_quat_mult(jnp.array(a), jnp.array(b)))
    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_quat_invert():
    q = _rand(5, 4)
    torch_out = torch_quat_invert(torch.from_numpy(q)).numpy()
    jax_out = np.array(jax_quat_invert(jnp.array(q)))
    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_quat_rotation():
    q = _rand(5, 4)
    p = _rand(5, 3, seed=43)
    torch_out = torch_quat_rotation(torch.from_numpy(q), torch.from_numpy(p)).numpy()
    jax_out = np.array(jax_quat_rotation(jnp.array(q), jnp.array(p)))
    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_sqrt_subgradient():
    x = np.array([-1.0, 0.0, 0.5, 2.0, -0.3], dtype=np.float32)
    torch_out = torch_sqrt_subgradient(torch.from_numpy(x)).numpy()
    jax_out = np.array(jax_sqrt_subgradient(jnp.array(x)))
    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_graham_schmidt():
    x_axis = _rand(5, 3)
    xy_plane = _rand(5, 3, seed=43)
    torch_out = torch_graham_schmidt(
        torch.from_numpy(x_axis), torch.from_numpy(xy_plane)
    ).numpy()
    jax_out = np.array(jax_graham_schmidt(jnp.array(x_axis), jnp.array(xy_plane)))
    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_rotation_quat_identity():
    shape = (3, 5)
    torch_out = TorchRotationQuat.identity(shape).tensor.numpy()
    jax_out = np.array(JaxRotationQuat.identity(shape).tensor)
    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_rotation_quat_compose():
    a = _rand(5, 4)
    b = _rand(5, 4, seed=43)
    torch_out = (
        TorchRotationQuat(torch.from_numpy(a))
        .compose(TorchRotationQuat(torch.from_numpy(b)))
        .tensor.numpy()
    )
    jax_out = np.array(
        JaxRotationQuat(jnp.array(a)).compose(JaxRotationQuat(jnp.array(b))).tensor
    )
    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_rotation_quat_apply():
    q = _rand(5, 4)
    p = _rand(5, 3, seed=43)
    torch_out = (
        TorchRotationQuat(torch.from_numpy(q), normalized=True)
        .apply(torch.from_numpy(p))
        .numpy()
    )
    jax_out = np.array(
        JaxRotationQuat(jnp.array(q), normalized=True).apply(jnp.array(p))
    )
    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_rotation_quat_invert():
    q = _rand(5, 4)
    torch_out = TorchRotationQuat(torch.from_numpy(q)).invert().tensor.numpy()
    jax_out = np.array(JaxRotationQuat(jnp.array(q)).invert().tensor)
    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_rotation_quat_to_matrix():
    q = _rand(5, 4)
    torch_out = (
        TorchRotationQuat(torch.from_numpy(q), normalized=True)
        .as_matrix()
        .to_3x3()
        .numpy()
    )
    jax_out = np.array(
        JaxRotationQuat(jnp.array(q), normalized=True).as_matrix().to_3x3()
    )
    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_rotation_matrix_identity():
    shape = (3, 5)
    torch_out = TorchRotationMatrix.identity(shape).to_3x3().numpy()
    jax_out = np.array(JaxRotationMatrix.identity(shape).to_3x3())
    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_rotation_matrix_compose():
    r1 = _rand(5, 3, 3)
    r2 = _rand(5, 3, 3, seed=43)
    torch_result = TorchRotationMatrix(torch.from_numpy(r1)).compose(
        TorchRotationMatrix(torch.from_numpy(r2))
    )
    jax_result = JaxRotationMatrix(jnp.array(r1)).compose(
        JaxRotationMatrix(jnp.array(r2))
    )
    assert np.allclose(
        torch_result.tensor.numpy(), np.array(jax_result.tensor), atol=1e-5
    )


def test_rotation_matrix_apply():
    r = _rand(5, 3, 3)
    p = _rand(5, 3, seed=43)
    torch_out = (
        TorchRotationMatrix(torch.from_numpy(r)).apply(torch.from_numpy(p)).numpy()
    )
    jax_out = np.array(JaxRotationMatrix(jnp.array(r)).apply(jnp.array(p)))
    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_rotation_matrix_invert():
    r = _rand(5, 3, 3)
    torch_out = TorchRotationMatrix(torch.from_numpy(r)).invert().tensor.numpy()
    jax_out = np.array(JaxRotationMatrix(jnp.array(r)).invert().tensor)
    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_rotation_matrix_to_quat():
    q = _rand(5, 4)
    torch_mat = TorchRotationQuat(torch.from_numpy(q), normalized=True).as_matrix()
    torch_out = torch_mat.as_quat().tensor.numpy()
    jax_mat = JaxRotationQuat(jnp.array(q), normalized=True).as_matrix()
    jax_out = np.array(jax_mat.as_quat().tensor)
    assert np.allclose(torch_out, jax_out, atol=1e-5)


@pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 3, 4)])
def test_affine3d_compose(shape):
    t1 = _rand(*shape, 3)
    r1 = _rand(*shape, 3, 3)
    t2 = _rand(*shape, 3, seed=43)
    r2 = _rand(*shape, 3, 3, seed=44)

    torch_a1 = TorchAffine3D(
        torch.from_numpy(t1), TorchRotationMatrix(torch.from_numpy(r1))
    )
    torch_a2 = TorchAffine3D(
        torch.from_numpy(t2), TorchRotationMatrix(torch.from_numpy(r2))
    )
    torch_out = torch_a1.compose(torch_a2).tensor.numpy()

    jax_a1 = JaxAffine3D(jnp.array(t1), JaxRotationMatrix(jnp.array(r1)))
    jax_a2 = JaxAffine3D(jnp.array(t2), JaxRotationMatrix(jnp.array(r2)))
    jax_out = np.array(jax_a1.compose(jax_a2).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-5)


@pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 3, 4)])
def test_affine3d_apply(shape):
    t = _rand(*shape, 3)
    r = _rand(*shape, 3, 3)
    p = _rand(*shape, 3, seed=43)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r))
    )
    torch_out = torch_a.apply(torch.from_numpy(p)).numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r)))
    jax_out = np.array(jax_a.apply(jnp.array(p)))

    assert np.allclose(torch_out, jax_out, atol=1e-5)


@pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 3, 4)])
def test_affine3d_invert(shape):
    t = _rand(*shape, 3)
    r = _rand(*shape, 3, 3)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r))
    )
    torch_out = torch_a.invert().tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r)))
    jax_out = np.array(jax_a.invert().tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-5)


@pytest.mark.parametrize("last_dim", [7, 12])
def test_affine3d_from_tensor(last_dim):
    if last_dim == 7:
        t = _rand(5, 7)
    else:
        t = _rand(5, 12)

    torch_a = TorchAffine3D.from_tensor(torch.from_numpy(t))
    torch_out = torch_a.tensor.numpy()

    jax_a = JaxAffine3D.from_tensor(jnp.array(t))
    jax_out = np.array(jax_a.tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_affine3d_from_graham_schmidt():
    neg_x = _rand(5, 3)
    origin = _rand(5, 3, seed=43)
    xy = _rand(5, 3, seed=44)

    torch_a = TorchAffine3D.from_graham_schmidt(
        torch.from_numpy(neg_x), torch.from_numpy(origin), torch.from_numpy(xy)
    )
    torch_out = torch_a.tensor.numpy()

    jax_a = JaxAffine3D.from_graham_schmidt(
        jnp.array(neg_x), jnp.array(origin), jnp.array(xy)
    )
    jax_out = np.array(jax_a.tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-5)


@pytest.mark.parametrize("shape", [(5,), (3, 5)])
def test_affine3d_scale(shape):
    t = _rand(*shape, 3)
    r = _rand(*shape, 3, 3)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r))
    )
    torch_out = torch_a.scale(2.5).tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r)))
    jax_out = np.array(jax_a.scale(2.5).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_affine3d_cat():
    t1 = _rand(5, 3)
    r1 = _rand(5, 3, 3)
    t2 = _rand(3, 3, seed=43)
    r2 = _rand(3, 3, 3, seed=44)

    torch_a1 = TorchAffine3D(
        torch.from_numpy(t1), TorchRotationMatrix(torch.from_numpy(r1))
    )
    torch_a2 = TorchAffine3D(
        torch.from_numpy(t2), TorchRotationMatrix(torch.from_numpy(r2))
    )
    torch_out = TorchAffine3D.cat([torch_a1, torch_a2], dim=0).tensor.numpy()

    jax_a1 = JaxAffine3D(jnp.array(t1), JaxRotationMatrix(jnp.array(r1)))
    jax_a2 = JaxAffine3D(jnp.array(t2), JaxRotationMatrix(jnp.array(r2)))
    jax_out = np.array(JaxAffine3D.cat([jax_a1, jax_a2], dim=0).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_affine3d_from_tensor_case6():
    t = _rand(5, 6)

    torch_a = TorchAffine3D.from_tensor(torch.from_numpy(t))
    torch_out = torch_a.tensor.numpy()

    jax_a = JaxAffine3D.from_tensor(jnp.array(t))
    jax_out = np.array(jax_a.tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


@pytest.mark.parametrize("shape", [(5,), (3, 5)])
def test_affine3d_identity_from_shape(shape):
    torch_out = TorchAffine3D.identity(shape).tensor.numpy()
    jax_out = np.array(JaxAffine3D.identity(shape).tensor)
    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_affine3d_identity_from_affine():
    t = _rand(5, 3)
    r = _rand(5, 3, 3)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r))
    )
    torch_out = TorchAffine3D.identity(torch_a).tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r)))
    jax_out = np.array(JaxAffine3D.identity(jax_a).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


@pytest.mark.parametrize("shape", [(5,), (3, 5)])
def test_affine3d_mask_with_zero(shape):
    t = _rand(*shape, 3)
    r = _rand(*shape, 3, 3)
    np.random.seed(99)
    m = np.random.choice([True, False], size=shape)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r))
    )
    torch_out = torch_a.mask(torch.from_numpy(m), with_zero=True).tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r)))
    jax_out = np.array(jax_a.mask(jnp.array(m), with_zero=True).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


@pytest.mark.parametrize("shape", [(5,), (3, 5)])
def test_affine3d_mask_with_identity(shape):
    t = _rand(*shape, 3)
    r = _rand(*shape, 3, 3)
    np.random.seed(99)
    m = np.random.choice([True, False], size=shape)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r))
    )
    torch_out = torch_a.mask(torch.from_numpy(m), with_zero=False).tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r)))
    jax_out = np.array(jax_a.mask(jnp.array(m), with_zero=False).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_convert_compose_quat_to_matrix():
    q = _rand(5, 4)
    r = _rand(5, 3, 3, seed=43)

    torch_rm = TorchRotationMatrix(torch.from_numpy(r))
    torch_rq = TorchRotationQuat(torch.from_numpy(q))
    torch_out = torch_rm.convert_compose(torch_rq).tensor.numpy()

    jax_rm = JaxRotationMatrix(jnp.array(r))
    jax_rq = JaxRotationQuat(jnp.array(q))
    jax_out = np.array(jax_rm.convert_compose(jax_rq).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_affine3d_compose_rotation():
    t = _rand(5, 3)
    r1 = _rand(5, 3, 3)
    r2 = _rand(5, 3, 3, seed=43)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r1))
    )
    torch_out = torch_a.compose_rotation(
        TorchRotationMatrix(torch.from_numpy(r2))
    ).tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r1)))
    jax_out = np.array(jax_a.compose_rotation(JaxRotationMatrix(jnp.array(r2))).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-5)


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4)])
def test_rotation_matrix_to_quat_roundtrip(shape):
    q = _rand(*shape, 4)
    torch_rq = TorchRotationQuat(torch.from_numpy(q), normalized=True)
    torch_out = torch_rq.as_matrix().as_quat().tensor.numpy()

    jax_rq = JaxRotationQuat(jnp.array(q), normalized=True)
    jax_out = np.array(jax_rq.as_matrix().as_quat().tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_affine3d_getitem():
    t = _rand(5, 3)
    r = _rand(5, 3, 3)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r))
    )
    torch_out = torch_a[2].tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r)))
    jax_out = np.array(jax_a[2].tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_rotation_quat_tensor_apply():
    q = _rand(5, 4)
    func = lambda x: x * 2.0 + 1.0

    torch_out = TorchRotationQuat(torch.from_numpy(q)).tensor_apply(func).tensor.numpy()
    jax_out = np.array(JaxRotationQuat(jnp.array(q)).tensor_apply(func).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_rotation_matrix_tensor_apply():
    r = _rand(5, 3, 3)
    func = lambda x: x * 0.5

    torch_out = (
        TorchRotationMatrix(torch.from_numpy(r)).tensor_apply(func).tensor.numpy()
    )
    jax_out = np.array(JaxRotationMatrix(jnp.array(r)).tensor_apply(func).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_affine3d_tensor_apply():
    t = _rand(5, 3)
    r = _rand(5, 3, 3)
    func = lambda x: x + 1.0

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r))
    )
    torch_out = torch_a.tensor_apply(func).tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r)))
    jax_out = np.array(jax_a.tensor_apply(func).tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_affine3d_as_matrix():
    q = _rand(5, 4)
    t = _rand(5, 3, seed=43)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationQuat(torch.from_numpy(q), normalized=True)
    )
    torch_out = torch_a.as_matrix().tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationQuat(jnp.array(q), normalized=True))
    jax_out = np.array(jax_a.as_matrix().tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_affine3d_as_quat():
    r = _rand(5, 3, 3)
    t = _rand(5, 3, seed=43)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r))
    )
    torch_out = torch_a.as_quat().tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r)))
    jax_out = np.array(jax_a.as_quat().tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-5)


def test_affine3d_getitem_none():
    t = _rand(5, 3)
    r = _rand(5, 3, 3)

    torch_a = TorchAffine3D(
        torch.from_numpy(t), TorchRotationMatrix(torch.from_numpy(r))
    )
    torch_out = torch_a[None].tensor.numpy()

    jax_a = JaxAffine3D(jnp.array(t), JaxRotationMatrix(jnp.array(r)))
    jax_out = np.array(jax_a[None].tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


def test_rotation_matrix_getitem_multidim():
    r = _rand(3, 5, 3, 3)

    torch_out = TorchRotationMatrix(torch.from_numpy(r))[1, 2:4].tensor.numpy()
    jax_out = np.array(JaxRotationMatrix(jnp.array(r))[1, 2:4].tensor)

    assert np.allclose(torch_out, jax_out, atol=1e-6)


@pytest.mark.parametrize(
    "seq_len, nan_ratio",
    [
        (7, 0.0),
        (11, 0.3),
        (11, 1.0),
    ],
)
def test_build_affine3d_from_coordinates(seq_len, nan_ratio):
    batch = 2
    np.random.seed(42)

    coords = np.random.randn(batch, seq_len, 3, 3).astype(np.float32)

    num_nan = int(seq_len * nan_ratio)
    if num_nan > 0:
        coords[:, :num_nan, :, :] = np.nan

    torch_coords = torch.from_numpy(coords.copy())
    torch_affine, torch_mask = torch_build_affine(torch_coords)
    torch_tensor = torch_affine.tensor.detach().numpy()
    torch_mask = torch_mask.numpy()

    jax_coords = jnp.array(coords)

    def call_single(c):
        affine, mask = jax_build_affine(c)
        return affine.tensor, mask

    jax_tensor, jax_mask = eqx.filter_vmap(call_single)(jax_coords)

    assert np.array_equal(torch_mask, np.array(jax_mask))
    assert np.allclose(torch_tensor, np.array(jax_tensor), atol=1e-5, equal_nan=True)
