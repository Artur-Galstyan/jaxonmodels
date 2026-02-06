import functools
import math
from abc import ABC

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import (
    Any,
    Literal,
    Self,
    Type,
    Union,
)
from jaxonlayers.functions.activation import swiglu
from jaxonlayers.functions.normalization import normalize
from jaxonlayers.layers.embedding import EmbeddingBag, EmbeddingWithPadding
from jaxtyping import Array, Float, Int, PRNGKeyArray

from jaxonmodels.functions import default_floating_dtype

ESM3_OPEN_SMALL = "esm3_sm_open_v1"
ESM3_OPEN_SMALL_ALIAS_1 = "esm3-open-2024-03"
ESM3_OPEN_SMALL_ALIAS_2 = "esm3-sm-open-v1"
ESM3_OPEN_SMALL_ALIAS_3 = "esm3-open"
ESM3_STRUCTURE_ENCODER_V0 = "esm3_structure_encoder_v0"
ESM3_STRUCTURE_DECODER_V0 = "esm3_structure_decoder_v0"
ESM3_FUNCTION_DECODER_V0 = "esm3_function_decoder_v0"
ESMC_600M = "esmc_600m"
ESMC_300M = "esmc_300m"


SEQUENCE_MASK_TOKEN = 32
SEQUENCE_BOS_TOKEN = 0
SEQUENCE_PAD_TOKEN = 1
SEQUENCE_EOS_TOKEN = 2
SEQUENCE_CHAINBREAK_TOKEN = 31
STRUCTURE_MASK_TOKEN = 4096
STRUCTURE_PAD_TOKEN = 4099
STRUCTURE_BOS_TOKEN = 4098
STRUCTURE_EOS_TOKEN = 4097
STRUCTURE_CHAINBREAK_TOKEN = 4100
SS8_PAD_TOKEN = 0
SASA_PAD_TOKEN = 0
RESIDUE_PAD_TOKEN = 0
INTERPRO_PAD_TOKEN = 0


def _graham_schmidt(x_axis: Array, xy_plane: Array, eps: float = 1e-12):
    e1 = xy_plane
    denom = jnp.sqrt((x_axis**2).sum(axis=-1, keepdims=True) + eps)
    x_axis = x_axis / denom
    dot = (x_axis * e1).sum(axis=-1, keepdims=True)
    e1 = e1 - x_axis * dot
    denom = jnp.sqrt((e1**2).sum(axis=-1, keepdims=True) + eps)
    e1 = e1 / denom
    e2 = jnp.cross(x_axis, e1, axis=-1)
    rots = jnp.stack([x_axis, e1, e2], axis=-1)
    return rots


def _sqrt_subgradient(x: Array) -> Array:
    return jnp.where(x > 0, jnp.sqrt(x), 0.0)


def _quat_invert(q: Array):
    return q * jnp.array([1, -1, -1, -1])


def _quat_rotation(q: Float[Array, "w x y z"], p: Float[Array, "x y z"]) -> Array:
    """
    Rotates p by quaternion q.

    Args:
        q: Quaternions as tensor of shape (..., 4), real part first.
        p: Points as tensor of shape (..., 3)

    Returns:
        The rotated version of p, of shape (..., 3)
    """
    aw, ax, ay, az = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    bx, by, bz = p[..., 0], p[..., 1], p[..., 2]
    # fmt: off
    ow =         - ax * bx - ay * by - az * bz
    ox = aw * bx           + ay * bz - az * by
    oy = aw * by - ax * bz           + az * bx
    oz = aw * bz + ax * by - ay * bx
    # fmt: on
    q_mul_pts = jnp.stack((ow, ox, oy, oz), -1)
    return _quat_mult(q_mul_pts, _quat_invert(q))[..., 1:]


def _quat_mult(
    a: Float[Array, "w x y z"], b: Float[Array, "w x y z"]
) -> Float[Array, "w x y z"]:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return jnp.stack((ow, ox, oy, oz), -1)


class Rotation(ABC):
    @classmethod
    def identity(cls, shape: tuple[int, ...], **tensor_kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    def random(cls, shape: tuple[int, ...], key: PRNGKeyArray, **tensor_kwargs) -> Self:
        raise NotImplementedError

    def __getitem__(self, idx: Any) -> Self:
        raise NotImplementedError

    @property
    def tensor(self) -> Array:
        raise NotImplementedError

    @property
    def shape(self) -> tuple:
        raise NotImplementedError

    def as_matrix(self) -> "RotationMatrix":
        raise NotImplementedError

    def as_quat(self, normalize: bool = False) -> "RotationQuat":
        raise NotImplementedError

    def compose(self, other: "Rotation") -> "Rotation":
        raise NotImplementedError

    def convert_compose(self, other: Self) -> Self:
        raise NotImplementedError

    def apply(self, p: Array) -> Array:
        raise NotImplementedError

    def invert(self) -> Self:
        raise NotImplementedError

    @property
    def dtype(self) -> jnp.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> jax.Device:
        return self.tensor.device

    def tensor_apply(self, func) -> Self:
        # Applys a function to the underlying tensor
        return type(self)(
            jnp.stack(
                [func(self.tensor[..., i]) for i in range(self.tensor.shape[-1])],
                axis=-1,
            )  # ty: ignore too-many-positional-arguments
        )


class RotationQuat(Rotation, eqx.Module):
    _quats: Array
    _normalized: bool = eqx.field(static=True)

    def __init__(self, quats: Array, normalized=False):
        assert quats.shape[-1] == 4
        self._normalized = normalized
        # Force float32 as well
        if normalized:
            self._quats = normalize(quats.astype(jnp.float32), axis=-1)
            self._quats = jnp.where(
                self._quats[..., :1] >= 0, self._quats, -self._quats
            )
        else:
            self._quats = quats.astype(jnp.float32)

    @classmethod
    def identity(cls, shape, **tensor_kwargs):
        q = jnp.ones((*shape, 4), **tensor_kwargs)
        mult = jnp.array([1, 0, 0, 0])
        return RotationQuat(q * mult)

    @classmethod
    def random(cls, shape, key: PRNGKeyArray, **tensor_kwargs):
        quat = jax.random.normal(key=key, shape=(*shape, 4), **tensor_kwargs)
        return RotationQuat(quat, normalized=True)

    def __getitem__(self, idx: Any) -> "RotationQuat":
        if isinstance(idx, (int, slice)) or idx is None:
            indices = (idx,)
        else:
            indices = tuple(idx)
        return RotationQuat(self._quats[indices + (slice(None),)])

    @property
    def shape(self) -> tuple:
        return self._quats.shape[:-1]

    def compose(self, other: Rotation) -> Rotation:
        assert isinstance(other, RotationQuat)
        return RotationQuat(_quat_mult(self._quats, other._quats))

    def convert_compose(self, other: Rotation):
        return self.compose(other.as_quat())

    def as_matrix(self) -> "RotationMatrix":
        q = self.normalized().tensor
        r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        two_s = 2.0 / jnp.linalg.norm(q, axis=-1)

        o = jnp.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return RotationMatrix(o.reshape(q.shape[:-1] + (3, 3)))

    def as_quat(self, normalize: bool = False) -> "RotationQuat":
        return self

    def apply(self, p: Array) -> Array:
        return _quat_rotation(self.normalized()._quats, p)

    def invert(self) -> "RotationQuat":
        return RotationQuat(_quat_invert(self._quats))

    @property
    def tensor(self) -> Array:
        return self._quats

    def normalized(self) -> "RotationQuat":
        return self if self._normalized else RotationQuat(self._quats, normalized=True)


class RotationMatrix(Rotation, eqx.Module):
    _rots: Array

    def __init__(self, rots: Array):
        if rots.shape[-1] == 9:
            rots = rots.reshape(*rots.shape[:-1], 3, 3)
        assert rots.shape[-1] == 3
        assert rots.shape[-2] == 3
        # Force full precision
        rots = rots.astype(jnp.float32)
        self._rots = rots

    @classmethod
    def identity(cls, shape, **tensor_kwargs):
        rots = jnp.eye(3, **tensor_kwargs)
        rots = rots.reshape(*[1 for _ in range(len(shape))], 3, 3)
        rots = jnp.broadcast_to(rots, (*shape, rots.shape[-2], rots.shape[-1]))
        return cls(rots)

    @classmethod
    def random(cls, shape, key: PRNGKeyArray, **tensor_kwargs):
        return RotationQuat.random(shape, **tensor_kwargs).as_matrix()

    def __getitem__(self, idx: Any) -> "RotationMatrix":
        indices = (idx,) if isinstance(idx, int) or idx is None else tuple(idx)
        return RotationMatrix(self._rots[indices + (slice(None), slice(None))])

    @property
    def shape(self) -> tuple:
        return self._rots.shape[:-2]

    def as_matrix(self) -> "RotationMatrix":
        return self

    def as_quat(self, normalize: bool = False) -> RotationQuat:
        flat = self._rots.reshape(*self._rots.shape[:-2], 9)
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = [flat[..., i] for i in range(9)]
        q_abs = _sqrt_subgradient(
            jnp.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                axis=-1,
            )
        )
        # we produce the desired quaternion multiplied by each of r, i, j, k
        quat_by_rijk = jnp.stack(
            [
                x
                for lst in [
                    [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01],
                    [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20],
                    [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21],
                    [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2],
                ]
                for x in lst
            ],
            axis=-1,
        )
        quat_by_rijk = quat_by_rijk.reshape(*quat_by_rijk.shape[:-1], 4, 4)

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = jnp.array(0.1, dtype=q_abs.dtype)
        quat_candidates = quat_by_rijk / (2.0 * jnp.maximum(q_abs[..., None], flr))

        one_hot = jax.nn.one_hot(q_abs.argmax(axis=-1), num_classes=q_abs.shape[-1])
        quat = (quat_candidates * one_hot[..., :, None]).sum(axis=-2)

        return RotationQuat(quat)

    def compose(self, other: Rotation) -> Rotation:
        assert isinstance(other, RotationMatrix)
        return RotationMatrix(self._rots @ other._rots)

    def convert_compose(self, other: Rotation):
        return self.compose(other.as_matrix())

    def apply(self, p: Array) -> Array:
        if self._rots.shape[-3] == 1:
            # This is a slight speedup over einsum for batched rotations
            return p @ jnp.swapaxes(self._rots, -1, -2).squeeze(-3)
        else:
            # einsum way faster than bmm!
            return jnp.einsum("...ij,...j", self._rots, p)

    def invert(self) -> "RotationMatrix":
        return RotationMatrix(jnp.swapaxes(self._rots, -1, -2))

    @property
    def tensor(self) -> Array:
        return self._rots.reshape(*self._rots.shape[:-2], -1)

    def to_3x3(self) -> Array:
        return self._rots

    @staticmethod
    def from_graham_schmidt(
        x_axis: Array, xy_plane: Array, eps: float = 1e-12
    ) -> "RotationMatrix":
        return RotationMatrix(_graham_schmidt(x_axis, xy_plane, eps))


class Affine3D(eqx.Module):
    trans: Array
    rot: Rotation

    def __post_init__(self):
        assert self.trans.shape[:-1] == self.rot.shape

    @staticmethod
    def identity(
        shape_or_affine: Union[tuple[int, ...], "Affine3D"],
        rotation_type: Type[Rotation] = RotationMatrix,
        **tensor_kwargs,
    ):
        if isinstance(shape_or_affine, Affine3D):
            shape = shape_or_affine.shape
            rotation_type = type(shape_or_affine.rot)
            dtype = shape_or_affine.dtype
        else:
            shape = shape_or_affine
            dtype = tensor_kwargs.get("dtype", None)

        return Affine3D(
            jnp.zeros((*shape, 3), dtype=dtype),
            rotation_type.identity(shape, dtype=dtype),
        )

    @staticmethod
    def random(
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        std: float = 1,
        rotation_type: Type[Rotation] = RotationMatrix,
        **tensor_kwargs,
    ) -> "Affine3D":
        return Affine3D(
            trans=jax.random.normal(key=key, shape=(*shape, 3), **tensor_kwargs) * std,
            rot=rotation_type.random(shape, **tensor_kwargs),
        )

    def __getitem__(self, idx: Any) -> "Affine3D":
        indices = (idx,) if isinstance(idx, int) or idx is None else tuple(idx)
        return Affine3D(trans=self.trans[indices + (slice(None),)], rot=self.rot[idx])

    @property
    def shape(self) -> tuple:
        return self.trans.shape[:-1]

    @property
    def dtype(self) -> jnp.dtype:
        return self.trans.dtype

    @property
    def device(self) -> jax.Device:
        return self.trans.device

    def tensor_apply(self, func) -> "Affine3D":
        # Applys a function to the underlying tensor
        return self.from_tensor(
            jnp.stack(
                [func(self.tensor[..., i]) for i in range(self.tensor.shape[-1])],
                axis=-1,
            )
        )

    def as_matrix(self):
        return Affine3D(trans=self.trans, rot=self.rot.as_matrix())

    def as_quat(self, normalize: bool = False):
        return Affine3D(trans=self.trans, rot=self.rot.as_quat(normalize))

    def compose(self, other: "Affine3D", autoconvert: bool = False):
        rot = self.rot
        new_rot = (rot.convert_compose if autoconvert else rot.compose)(other.rot)
        new_trans = rot.apply(other.trans) + self.trans
        return Affine3D(trans=new_trans, rot=new_rot)

    def compose_rotation(self, other: Rotation, autoconvert: bool = False):
        return Affine3D(
            trans=self.trans,
            rot=(self.rot.convert_compose if autoconvert else self.rot.compose)(other),
        )

    def scale(self, v: Array | float):
        return Affine3D(self.trans * v, self.rot)

    def mask(self, mask: Array, with_zero=False):
        # Returns a transform where True positions in mask is identity
        if with_zero:
            tensor = self.tensor
            return Affine3D.from_tensor(
                jnp.where(mask[..., None], jnp.zeros_like(tensor), tensor)
            )
        else:
            identity = self.identity(
                self.shape,
                rotation_type=type(self.rot),
                dtype=self.dtype,
            ).tensor
            return Affine3D.from_tensor(
                jnp.where(mask[..., None], identity, self.tensor)
            )

    def apply(self, p: Array) -> Array:
        return self.rot.apply(p) + self.trans

    def invert(self):
        inv_rot = self.rot.invert()
        return Affine3D(trans=-inv_rot.apply(self.trans), rot=inv_rot)

    @property
    def tensor(self) -> Array:
        return jnp.concat([self.rot.tensor, self.trans], axis=-1)

    @staticmethod
    def from_tensor(t: Array) -> "Affine3D":
        match t.shape[-1]:
            case 4:
                # Assume tensor 4x4 for backward compat with alphafold
                trans = t[..., :3, 3]
                rot = RotationMatrix(t[..., :3, :3])
            case 6:
                trans = t[..., -3:]
                x = t[..., :3]
                padded = jnp.concatenate([jnp.ones((*x.shape[:-1], 1)), x], axis=-1)
                rot = RotationQuat(padded)
            case 7:
                trans = t[..., -3:]
                rot = RotationQuat(t[..., :4])
            case 12:
                trans = t[..., -3:]
                rot = RotationMatrix(t[..., :-3].reshape(*t.shape[:-1], 3, 3))
            case _:
                raise RuntimeError(
                    "Cannot detect rotation fromat "
                    f"from {t.shape[-1] - 3}-d flat vector"
                )
        return Affine3D(trans, rot)

    @staticmethod
    def from_tensor_pair(t: Array, r: Array) -> "Affine3D":
        return Affine3D(t, RotationMatrix(r))

    @staticmethod
    def from_graham_schmidt(
        neg_x_axis: Array,
        origin: Array,
        xy_plane: Array,
        eps: float = 1e-10,
    ):
        # The arguments of this function is for parity with AlphaFold
        x_axis = origin - neg_x_axis
        xy_plane = xy_plane - origin
        return Affine3D(
            trans=origin, rot=RotationMatrix.from_graham_schmidt(x_axis, xy_plane, eps)
        )

    @staticmethod
    def cat(affines: list["Affine3D"], dim: int = 0):
        if dim < 0:
            dim = len(affines[0].shape) + dim
        return Affine3D.from_tensor(jnp.concat([x.tensor for x in affines], axis=dim))


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


def swiglu_ln_ffn(
    d_model: int,
    expansion_ratio: float,
    bias: bool,
    *,
    key: PRNGKeyArray,
    dtype: Any | None,
):
    key1, key2 = jax.random.split(key)
    return [
        eqx.nn.LayerNorm(d_model),
        eqx.nn.Linear(
            d_model,
            swiglu_correction_fn(expansion_ratio, d_model) * 2,
            use_bias=bias,
            key=key1,
            dtype=dtype,
        ),
        eqx.nn.Lambda(fn=swiglu),
        eqx.nn.Linear(
            swiglu_correction_fn(expansion_ratio, d_model),
            d_model,
            use_bias=bias,
            key=key2,
            dtype=dtype,
        ),
    ]


def gelu_ln_ffn(
    d_model: int,
    expansion_ratio: float,
    bias: bool,
    *,
    key: PRNGKeyArray,
    dtype: Any | None,
):
    hidden_dim = int(expansion_ratio * d_model)
    key1, key2 = jax.random.split(key)
    return [
        eqx.nn.LayerNorm(d_model),
        eqx.nn.Linear(d_model, hidden_dim, use_bias=bias, key=key1, dtype=dtype),
        eqx.nn.Lambda(fn=jax.nn.gelu),
        eqx.nn.Linear(hidden_dim, d_model, use_bias=bias, key=key2, dtype=dtype),
    ]


class GeometricReasoningOriginalImpl(eqx.Module):
    s_norm: eqx.nn.LayerNorm
    proj: eqx.nn.Linear
    distance_scale_per_head: Array
    rotation_scale_per_head: Array
    out_proj: eqx.nn.Linear

    v_heads: int = eqx.field(static=True)
    num_vector_messages: int = eqx.field(static=True)
    mask_and_zero_frameless: bool = eqx.field(static=True)

    inference: bool

    def __init__(
        self,
        c_s: int,
        v_heads: int,
        num_vector_messages: int = 1,
        mask_and_zero_frameless: bool = True,
        bias: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference

        self.v_heads = v_heads
        self.num_vector_messages = num_vector_messages
        self.mask_and_zero_frameless = mask_and_zero_frameless

        self.s_norm = eqx.nn.LayerNorm(c_s, use_bias=bias)
        dim_proj = 4 * self.v_heads * 3 + self.v_heads * 3 * self.num_vector_messages
        # 2 x (q, k) * number of heads * (x, y, z) +
        #   number of heads * number of vector messages * (x, y, z)

        key, proj_key, out_proj_key = jax.random.split(key, 3)
        self.proj = eqx.nn.Linear(
            c_s, dim_proj, use_bias=bias, key=proj_key, dtype=dtype
        )
        channels_out = self.v_heads * 3 * self.num_vector_messages
        self.out_proj = eqx.nn.Linear(
            channels_out, c_s, key=out_proj_key, use_bias=bias, dtype=dtype
        )
        self.distance_scale_per_head = jnp.zeros((self.v_heads))
        self.rotation_scale_per_head = jnp.zeros((self.v_heads))

    def __call__(
        self,
        s: Array,
        affine: Affine3D,
        affine_mask: Array,
        sequence_id: Array | None,
        chain_id: Array,
    ):
        seq_len = s.shape[0]

        if sequence_id is None:
            if jax.config.read("jax_enable_x64"):
                dtype = jnp.int64
            else:
                dtype = jnp.int32
            sequence_id = jnp.zeros(seq_len, dtype=dtype)

        attn_bias = sequence_id[:, None] == sequence_id[None, :]
        attn_bias = attn_bias[None, :, :].astype(s.dtype)
        attn_bias = jnp.where(
            affine_mask[None, None, :],
            attn_bias,
            jnp.finfo(attn_bias.dtype).min,
        )

        chain_id_mask = chain_id[:, None] != chain_id[None, :]
        attn_bias = jnp.where(
            chain_id_mask[None, :, :],
            jnp.finfo(s.dtype).min,
            attn_bias,
        )
        ns = eqx.filter_vmap(self.s_norm)(s)
        proj_out = eqx.filter_vmap(self.proj)(ns)

        split_idx = self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages
        vec_rot = proj_out[..., :split_idx]
        vec_dist = proj_out[..., split_idx:]

        vec_rot = vec_rot.reshape(seq_len, -1, 3)
        rotated = affine.rot[..., None].apply(vec_rot)

        qr_end = self.v_heads
        kr_end = self.v_heads * 2
        query_rot = rotated[:, :qr_end, :]
        key_rot = rotated[:, qr_end:kr_end, :]
        value = rotated[:, kr_end:, :]

        vec_dist = vec_dist.reshape(seq_len, -1, 3)
        transformed = affine[..., None].apply(vec_dist)
        query_dist, key_dist = jnp.split(transformed, 2, axis=-2)

        query_dist = jnp.transpose(query_dist, (1, 0, 2))[:, :, None, :]
        key_dist = jnp.transpose(key_dist, (1, 0, 2))[:, None, :, :]
        query_rot = jnp.transpose(query_rot, (1, 0, 2))
        key_rot = jnp.transpose(key_rot, (1, 2, 0))

        value = value.reshape(seq_len, self.v_heads, self.num_vector_messages, 3)
        value = jnp.transpose(value, (1, 0, 2, 3))
        value = value.reshape(self.v_heads, seq_len, self.num_vector_messages * 3)

        distance_term = jnp.linalg.norm(query_dist - key_dist, axis=-1) / math.sqrt(3)
        rotation_term = query_rot @ key_rot / math.sqrt(3)

        distance_term_weight = jax.nn.softplus(self.distance_scale_per_head)[
            :, None, None
        ]
        rotation_term_weight = jax.nn.softplus(self.rotation_scale_per_head)[
            :, None, None
        ]

        attn_weight = (
            rotation_term * rotation_term_weight - distance_term * distance_term_weight
        )

        if attn_bias is not None:
            s_q = attn_weight.shape[1]
            s_k = attn_weight.shape[2]
            _s_q = max(0, attn_bias.shape[1] - s_q)
            _s_k = max(0, attn_bias.shape[2] - s_k)
            attn_bias = attn_bias[:, _s_q:, _s_k:]
            attn_weight = attn_weight + attn_bias

        attn_weight = jnp.nan_to_num(jax.nn.softmax(attn_weight, axis=-1), nan=0.0)

        attn_out = attn_weight @ value

        attn_out = attn_out.reshape(self.v_heads, seq_len, self.num_vector_messages, 3)
        attn_out = jnp.transpose(attn_out, (1, 0, 2, 3))
        attn_out = attn_out.reshape(seq_len, self.v_heads * self.num_vector_messages, 3)

        attn_out = affine.rot[..., None].invert().apply(attn_out)

        attn_out = attn_out.reshape(
            seq_len, self.v_heads * self.num_vector_messages * 3
        )

        if self.mask_and_zero_frameless:
            attn_out = jnp.where(affine_mask[:, None], attn_out, 0.0)

        s = eqx.filter_vmap(self.out_proj)(attn_out)
        return s


class ESMMultiHeadAttention(eqx.Module):
    d_model: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    qk_layernorm: bool = eqx.field(static=True)

    layernorm_qkv: list

    q_ln: eqx.nn.LayerNorm | None
    k_ln: eqx.nn.LayerNorm | None
    rotary: eqx.nn.RotaryPositionalEmbedding
    out_proj: eqx.nn.Linear

    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_bias: bool = False,
        qk_layernorm: bool = True,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qk_layernorm = qk_layernorm

        key, *subkeys = jax.random.split(key, 3)

        self.layernorm_qkv = [
            eqx.nn.LayerNorm(d_model),
            eqx.nn.Linear(
                d_model, d_model * 3, use_bias=use_bias, key=subkeys[0], dtype=dtype
            ),
        ]

        if qk_layernorm:
            self.q_ln = eqx.nn.LayerNorm(d_model, use_bias=use_bias)
            self.k_ln = eqx.nn.LayerNorm(d_model, use_bias=use_bias)
        else:
            self.q_ln = None
            self.k_ln = None

        self.rotary = eqx.nn.RotaryPositionalEmbedding(self.d_head)
        self.out_proj = eqx.nn.Linear(
            d_model, d_model, use_bias=use_bias, key=subkeys[1], dtype=dtype
        )

    def __call__(
        self,
        x: Float[Array, "seq_len d_model"],
        seq_id: Int[Array, "seq_len"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len d_model"]:
        seq_len = x.shape[0]
        dtype = x.dtype

        for l in self.layernorm_qkv:
            x = eqx.filter_vmap(l)(x)
        qkv = x
        q, k, v = jnp.split(qkv, 3, axis=-1)

        if self.q_ln is not None:
            q = eqx.filter_vmap(self.q_ln)(q).astype(dtype)
        if self.k_ln is not None:
            k = eqx.filter_vmap(self.k_ln)(k).astype(dtype)

        q = q.reshape(seq_len, self.n_heads, self.d_head)
        k = k.reshape(seq_len, self.n_heads, self.d_head)
        v = v.reshape(seq_len, self.n_heads, self.d_head)

        q = eqx.filter_vmap(self.rotary, in_axes=1, out_axes=1)(q)
        k = eqx.filter_vmap(self.rotary, in_axes=1, out_axes=1)(k)

        mask = None
        if seq_id is not None:
            mask = seq_id[:, None] == seq_id[None, :]
            mask = jnp.expand_dims(mask, axis=0)
        context = jax.nn.dot_product_attention(q, k, v, mask=mask)
        context = context.reshape(seq_len, self.d_model)

        output = eqx.filter_vmap(self.out_proj)(context)

        return output


def RegressionHead(
    d_model: int,
    output_dim: int,
    hidden_dim: int | None = None,
    *,
    key: PRNGKeyArray,
    dtype: Any | None = None,
) -> list:
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    key, subkey = jax.random.split(key)
    return [
        eqx.nn.Linear(d_model, hidden_dim, key=key),
        eqx.nn.Lambda(fn=jax.nn.gelu),
        eqx.nn.LayerNorm(hidden_dim),
        eqx.nn.Linear(hidden_dim, output_dim, key=subkey),
    ]


class UnifiedTransformerBlock(eqx.Module):
    """
    A unified transformer block that can optionally incorporate geometric attention.

    This class defines a transformer block that can be configured to use
    geometric attention alongside the standard multi-head attention mechanism.
    It is designed to be a flexible component of transformer-based models, allowing for
    the integration of geometric reasoning.

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    n_layers : int
        The number of layers in the transformer block.
    use_geom_attn : bool, optional
        Whether to use geometric attention in addition to
        the standard multi-head attention. Defaults to False.
    v_heads : int, optional
        The number of heads to use for the geometric attention mechanism, if enabled.
        Must be specified if `use_geom_attn` is True.
    """

    attn: ESMMultiHeadAttention | None
    geom_attn: GeometricReasoningOriginalImpl | None
    ffn: list

    scaling_factor: float = eqx.field(static=True)

    use_plain_attn: bool = eqx.field(static=True)
    use_geom_attn: bool = eqx.field(static=True)

    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_geom_attn: bool = False,
        use_plain_attn: bool = True,
        v_heads: int | None = None,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        mask_and_zero_frameless: bool = False,
        qk_layernorm: bool = True,
        ffn_type: Literal["swiglu", "gelu"] = "swiglu",  # swiglu | gelu
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference
        self.use_plain_attn = use_plain_attn
        self.use_geom_attn = use_geom_attn
        key, mha_key = jax.random.split(key)
        if self.use_plain_attn:
            self.attn = ESMMultiHeadAttention(
                d_model,
                n_heads,
                bias,
                qk_layernorm=qk_layernorm,
                key=mha_key,
                inference=inference,
                dtype=dtype,
            )
        else:
            self.attn = None

        if self.use_geom_attn:
            if v_heads is None:
                raise ValueError("v_heads must be specified when use_geom_attn is True")
            key, geom_key = jax.random.split(key)
            self.geom_attn = GeometricReasoningOriginalImpl(
                c_s=d_model,
                v_heads=v_heads,
                bias=bias,
                mask_and_zero_frameless=mask_and_zero_frameless,
                key=geom_key,
                dtype=dtype,
                inference=inference,
            )
        else:
            self.geom_attn = None
        key, ffn_key = jax.random.split(key)
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(
                d_model,
                expansion_ratio,
                bias,
                key=ffn_key,
                dtype=dtype,
            )
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(
                d_model,
                expansion_ratio,
                bias,
                key=ffn_key,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.scaling_factor = residue_scaling_factor

    def __call__(
        self,
        x: Array,
        sequence_id: Array | None,
        frames: Affine3D | None,
        frames_mask: Array | None,
        chain_id: Array | None,
    ) -> Array:
        if self.use_plain_attn:
            assert self.attn is not None
            r1 = self.attn(x, sequence_id)
            x = x + r1 / self.scaling_factor

        if self.use_geom_attn:
            assert frames is not None
            assert frames_mask is not None
            assert self.geom_attn is not None
            assert chain_id is not None
            r2 = self.geom_attn(x, frames, frames_mask, sequence_id, chain_id)
            x = x + r2 / self.scaling_factor

        ffn_x = x
        for l in self.ffn:
            ffn_x = eqx.filter_vmap(l)(ffn_x)
        r3 = ffn_x / self.scaling_factor
        x = x + r3

        return x


class TransformerStack(eqx.Module):
    """
    A stack of transformer blocks used in the ESM-3 model.
    Each block is a UnifiedTransformerBlock,
    which can either be geometric attention or standard multi-head attention.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads.
        v_heads (int): The number of voting heads.
        n_layers (int): The number of transformer blocks in the stack.
        n_layers_geom (int, optional): The number of transformer blocks that use
            geometric attention.
        scale_residue (bool, optional): Whether to scale the residue connections
            in each transformer block.
        mask_and_zero_frameless (bool, optional): Whether to mask and zero frameless
            positions in the input.

        Only applies in the geometric attention blocks, which is
        conditioned on the structure
    """

    blocks: list[UnifiedTransformerBlock]
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int | None,
        n_layers: int,
        n_layers_geom: int = 1,
        scale_residue: bool = True,
        mask_and_zero_frameless: bool = False,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: Literal["swiglu", "gelu"] = "swiglu",  # swiglu | gelu
        expansion_ratio: float = 8 / 3,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if dtype is None:
            dtype = default_floating_dtype()
        assert dtype is not None
        keys = jax.random.split(key, n_layers)
        self.blocks = [
            UnifiedTransformerBlock(
                d_model,
                n_heads,
                v_heads=v_heads,
                use_geom_attn=i < n_layers_geom,
                residue_scaling_factor=(
                    math.sqrt(n_layers / 36) if scale_residue else 1.0
                ),
                expansion_ratio=expansion_ratio,
                mask_and_zero_frameless=mask_and_zero_frameless,
                bias=bias,
                qk_layernorm=qk_layernorm,
                ffn_type=ffn_type,
                key=keys[i],
                dtype=dtype,
                inference=inference,
            )
            for i in range(n_layers)
        ]
        self.norm = eqx.nn.LayerNorm(d_model, use_bias=False, dtype=dtype)

    def __call__(
        self,
        x: Array,
        sequence_id: Array | None = None,
        frames: Affine3D | None = None,
        frames_mask: Array | None = None,
        chain_id: Array | None = None,
    ) -> tuple[Array, Array, list[Array]]:
        hiddens = []
        for block in self.blocks:
            x = block(x, sequence_id, frames, frames_mask, chain_id)
            hiddens.append(x)
        embedding = x
        x = eqx.filter_vmap(self.norm)(x)
        return x, embedding, hiddens


class ESMC(eqx.Module):
    """
    ESMC model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        n_layers (int): The number of transformer layers.
    """

    embed: eqx.nn.Embedding
    transformer: TransformerStack
    sequence_head: list

    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if not dtype:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference
        key, embed_key, transformer_stack_key, regression_head_key = jax.random.split(
            key, 4
        )
        self.embed = eqx.nn.Embedding(64, d_model, key=embed_key)

        self.transformer = TransformerStack(
            d_model,
            n_heads,
            None,
            n_layers,
            n_layers_geom=0,
            key=transformer_stack_key,
            dtype=dtype,
            inference=inference,
        )

        self.sequence_head = RegressionHead(d_model, 64, key=regression_head_key)

    def __call__(
        self,
        sequence_tokens: Array,
        sequence_id: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        if sequence_id is None:
            # 1 is the PAD_TOKEN_ID
            sequence_id = sequence_tokens != 1

        x = eqx.filter_vmap(self.embed)(sequence_tokens)
        x, _, hiddens = self.transformer(x, sequence_id=sequence_id)
        hiddens = jnp.stack(hiddens, axis=0)
        sequence_logits = x
        for l in self.sequence_head:
            sequence_logits = eqx.filter_vmap(l)(sequence_logits)

        return sequence_logits, x, hiddens


def rbf(values, v_min, v_max, n_bins=16):
    rbf_centers = jnp.linspace(v_min, v_max, n_bins, dtype=values.dtype)
    rbf_centers = rbf_centers.reshape([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (jnp.expand_dims(values, axis=-1) - rbf_centers) / rbf_std
    return jnp.exp(-(z**2))


def build_affine3d_from_coordinates(
    coords: Array,
) -> tuple[Affine3D, Array]:
    _MAX_SUPPORTED_DISTANCE = 1e6
    coord_mask = jnp.all(
        jnp.all(jnp.isfinite(coords) & (coords < _MAX_SUPPORTED_DISTANCE), axis=-1),
        axis=-1,
    )

    def atom3_to_backbone_affine(bb_positions: Array) -> Affine3D:
        N = bb_positions[..., 0, :]
        CA = bb_positions[..., 1, :]
        C = bb_positions[..., 2, :]
        return Affine3D.from_graham_schmidt(C, CA, N)

    coords = coords.astype(jnp.float32)
    coords = jnp.where(coord_mask[:, None, None], coords, 0.0)

    masked_coords = jnp.where(coord_mask[:, None, None], coords, 0.0)
    average_per_n_ca_c = masked_coords.sum(axis=0) / (coord_mask.sum() + 1e-8)

    affine_from_average = atom3_to_backbone_affine(average_per_n_ca_c).as_matrix()

    S = coords.shape[0]

    affine_rot_mats = jnp.broadcast_to(affine_from_average.rot.tensor[None, :], (S, 9))
    affine_trans = jnp.broadcast_to(affine_from_average.trans[None, :], (S, 3))

    identity_rot = jax.lax.stop_gradient(
        RotationMatrix.identity((S,), dtype=jnp.float32)
    )

    has_any_coords = coord_mask.any()
    affine_rot_mats = jnp.where(has_any_coords, affine_rot_mats, identity_rot.tensor)

    black_hole_affine = Affine3D(affine_trans, RotationMatrix(affine_rot_mats))

    affine = atom3_to_backbone_affine(coords)
    affine = Affine3D.from_tensor(
        jnp.where(coord_mask[:, None], affine.tensor, black_hole_affine.tensor)
    )

    return affine, coord_mask


class OutputHeads(eqx.Module):
    sequence_head: list
    structure_head: list
    ss8_head: list
    sasa_head: list
    function_head: list
    residue_head: list

    def __init__(
        self,
        d_model: int,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if not dtype:
            dtype = default_floating_dtype()
        assert dtype is not None
        keys = jax.random.split(key, 6)
        self.sequence_head = RegressionHead(d_model, 64, key=keys[0], dtype=dtype)
        self.structure_head = RegressionHead(d_model, 4096, key=keys[1], dtype=dtype)
        self.ss8_head = RegressionHead(d_model, 8 + 3, key=keys[2], dtype=dtype)
        self.sasa_head = RegressionHead(d_model, 16 + 3, key=keys[3], dtype=dtype)
        self.function_head = RegressionHead(d_model, 260 * 8, key=keys[4], dtype=dtype)
        self.residue_head = RegressionHead(d_model, 1478, key=keys[5], dtype=dtype)

    def _apply_head(self, head: list, x: Array) -> Array:
        for layer in head:
            x = eqx.filter_vmap(layer)(x)
        return x

    def __call__(
        self, x: Array, embed: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        sequence_logits = self._apply_head(self.sequence_head, x)
        structure_logits = self._apply_head(self.structure_head, x)
        secondary_structure_logits = self._apply_head(self.ss8_head, x)
        sasa_logits = self._apply_head(self.sasa_head, x)
        function_logits = self._apply_head(self.function_head, x)
        function_logits = function_logits.reshape(*function_logits.shape[:-1], 8, -1)
        residue_logits = self._apply_head(self.residue_head, x)
        return (
            sequence_logits,
            structure_logits,
            secondary_structure_logits,
            sasa_logits,
            function_logits,
            residue_logits,
            embed,
        )


class EncodeInputs(eqx.Module):
    """
    Module for encoding input features in the ESM-3 model.

    Args:
        d_model (int): The dimensionality of the model's hidden states.
    """

    sequence_embed: eqx.nn.Embedding
    plddt_projection: eqx.nn.Linear
    structure_per_res_plddt_projection: eqx.nn.Linear
    structure_tokens_embed: eqx.nn.Embedding
    ss8_embed: eqx.nn.Embedding
    sasa_embed: eqx.nn.Embedding
    function_embed: list[EmbeddingWithPadding]
    residue_embed: EmbeddingBag

    def __init__(
        self,
        d_model: int,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if not dtype:
            dtype = default_floating_dtype()
        assert dtype is not None

        key, seq_embed_key = jax.random.split(key)
        self.sequence_embed = eqx.nn.Embedding(
            64, d_model, key=seq_embed_key, dtype=dtype
        )
        key, plddt_projection_key, struc_key = jax.random.split(key, 3)
        self.plddt_projection = eqx.nn.Linear(
            16, d_model, key=plddt_projection_key, dtype=dtype
        )
        self.structure_per_res_plddt_projection = eqx.nn.Linear(
            16, d_model, key=struc_key, dtype=dtype
        )

        key, struc_embed_key = jax.random.split(key)
        self.structure_tokens_embed = eqx.nn.Embedding(
            4096 + 5, d_model, key=struc_embed_key, dtype=dtype
        )

        key, ss8_key, sasa_key = jax.random.split(key, 3)
        # "Structural" features
        self.ss8_embed = eqx.nn.Embedding(8 + 3, d_model, key=ss8_key, dtype=dtype)
        self.sasa_embed = eqx.nn.Embedding(16 + 3, d_model, key=sasa_key, dtype=dtype)

        # "Functional" features
        key, *func_keys = jax.random.split(key, 9)
        self.function_embed = [
            EmbeddingWithPadding(
                260, d_model // 8, padding_idx=0, key=func_keys[i], dtype=dtype
            )
            for i in range(8)
        ]

        key, residue_key = jax.random.split(key)
        self.residue_embed = EmbeddingBag(
            1478, d_model, padding_idx=0, key=residue_key, dtype=dtype
        )

    def __call__(
        self,
        sequence_tokens: Array,
        structure_tokens: Array,
        average_plddt: Array,
        per_res_plddt: Array,
        ss8_tokens: Array,
        sasa_tokens: Array,
        function_tokens: Array,
        residue_annotation_tokens: Array,
    ) -> Array:
        sequence_embed = eqx.filter_vmap(self.sequence_embed)(sequence_tokens)

        rbf_16_fn = functools.partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt))
        structure_per_res_plddt = eqx.filter_vmap(
            self.structure_per_res_plddt_projection
        )(rbf_16_fn(per_res_plddt))

        structure_embed = eqx.filter_vmap(self.structure_tokens_embed)(structure_tokens)
        ss8_embed = eqx.filter_vmap(self.ss8_embed)(ss8_tokens)
        sasa_embed = eqx.filter_vmap(self.sasa_embed)(sasa_tokens)

        function_embed = jnp.concatenate(
            [
                eqx.filter_vmap(embed_fn)(function_tokens[..., i])
                for i, embed_fn in enumerate(self.function_embed)
            ],
            axis=-1,
        )

        residue_embed = eqx.filter_vmap(self.residue_embed)(residue_annotation_tokens)

        return (
            sequence_embed
            + plddt_embed
            + structure_per_res_plddt
            + structure_embed
            + ss8_embed
            + sasa_embed
            + function_embed
            + residue_embed
        )


class ESM3(eqx.Module):
    encoder: EncodeInputs
    transformer: TransformerStack
    output_heads: OutputHeads

    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int,
        n_layers: int,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        if not dtype:
            dtype = default_floating_dtype()
        assert dtype is not None
        self.inference = inference

        key, enc_key, trans_key, head_key = jax.random.split(key, 4)
        self.encoder = EncodeInputs(d_model, key=enc_key, dtype=dtype)
        self.transformer = TransformerStack(
            d_model,
            n_heads,
            v_heads,
            n_layers,
            mask_and_zero_frameless=True,
            key=trans_key,
            dtype=dtype,
            inference=inference,
        )
        self.output_heads = OutputHeads(d_model, key=head_key, dtype=dtype)

    def __call__(
        self,
        sequence_tokens: Array | None = None,
        structure_tokens: Array | None = None,
        ss8_tokens: Array | None = None,
        sasa_tokens: Array | None = None,
        function_tokens: Array | None = None,
        residue_annotation_tokens: Array | None = None,
        average_plddt: Array | None = None,
        per_res_plddt: Array | None = None,
        structure_coords: Array | None = None,
        chain_id: Array | None = None,
        sequence_id: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:

        L = next(
            x.shape[0]
            for x in [
                sequence_tokens,
                structure_tokens,
                ss8_tokens,
                sasa_tokens,
                structure_coords,
                function_tokens,
                residue_annotation_tokens,
            ]
            if x is not None
        )

        if sequence_tokens is None:
            sequence_tokens = jnp.full((L,), SEQUENCE_MASK_TOKEN, dtype=jnp.int32)
        if ss8_tokens is None:
            ss8_tokens = jnp.full((L,), SS8_PAD_TOKEN, dtype=jnp.int32)
        if sasa_tokens is None:
            sasa_tokens = jnp.full((L,), SASA_PAD_TOKEN, dtype=jnp.int32)
        if average_plddt is None:
            average_plddt = jnp.float32(1.0)
        if per_res_plddt is None:
            per_res_plddt = jnp.zeros((L,), dtype=jnp.float32)
        if chain_id is None:
            chain_id = jnp.zeros((L,), dtype=jnp.int32)
        if residue_annotation_tokens is None:
            residue_annotation_tokens = jnp.full(
                (L, 16), RESIDUE_PAD_TOKEN, dtype=jnp.int32
            )
        if function_tokens is None:
            function_tokens = jnp.full((L, 8), INTERPRO_PAD_TOKEN, dtype=jnp.int32)
        if structure_coords is None:
            structure_coords = jnp.full((L, 3, 3), jnp.nan, dtype=jnp.float32)

        structure_coords = structure_coords[..., :3, :]
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        if structure_tokens is None:
            structure_tokens = jnp.full((L,), STRUCTURE_MASK_TOKEN, dtype=jnp.int32)

        structure_tokens = jnp.where(
            structure_tokens == -1, STRUCTURE_MASK_TOKEN, structure_tokens
        )
        structure_tokens = jnp.where(
            sequence_tokens == SEQUENCE_BOS_TOKEN, STRUCTURE_BOS_TOKEN, structure_tokens
        )
        structure_tokens = jnp.where(
            sequence_tokens == SEQUENCE_PAD_TOKEN, STRUCTURE_PAD_TOKEN, structure_tokens
        )
        structure_tokens = jnp.where(
            sequence_tokens == SEQUENCE_EOS_TOKEN, STRUCTURE_EOS_TOKEN, structure_tokens
        )
        structure_tokens = jnp.where(
            sequence_tokens == SEQUENCE_CHAINBREAK_TOKEN,
            STRUCTURE_CHAINBREAK_TOKEN,
            structure_tokens,
        )

        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )

        x, embedding, _ = self.transformer(
            x, sequence_id, affine, affine_mask, chain_id
        )

        return self.output_heads(x, embedding)
