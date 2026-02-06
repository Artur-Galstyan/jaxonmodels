import math
from abc import ABC
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import Any, Literal, Self, Type, Union
from jaxonlayers.functions.activation import swiglu
from jaxonlayers.functions.normalization import normalize
from jaxtyping import Array, Float, Int, PRNGKeyArray

from jaxonmodels.functions import default_floating_dtype


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


class RotationQuat(Rotation):
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


class RotationMatrix(Rotation):
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


@dataclass(frozen=True)
class Affine3D:
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
    def __init__(
        self,
        c_s: int,
        v_heads: int,
        num_vector_messages: int = 1,
        mask_and_zero_frameless: bool = True,
        divide_residual_by_depth: bool = False,
        bias: bool = False,
        *,
        key: PRNGKeyArray,
        dtype: Any | None = None,
        inference: bool = False,
    ):
        pass


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
) -> eqx.nn.Sequential:
    """Single-hidden layer MLP for supervised output.

    Args:
        d_model: input dimension
        output_dim: dimensionality of the output.
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
    Returns:
        output MLP module.
    """
    if dtype is None:
        dtype = default_floating_dtype()
    assert dtype is not None
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    key, subkey = jax.random.split(key)
    return eqx.nn.Sequential(
        [
            eqx.nn.Linear(d_model, hidden_dim, key=key),
            eqx.nn.Lambda(fn=jax.nn.gelu),
            eqx.nn.LayerNorm(hidden_dim),
            eqx.nn.Linear(hidden_dim, output_dim, key=subkey),
        ]
    )


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

    attn: ESMMultiHeadAttention
    geom_attn: GeometricReasoningOriginalImpl
    ffn: list

    use_plain_attn: bool = eqx.field(static=True)
    use_geom_attn: bool = eqx.field(static=True)
    residue_scaling_factor: float = eqx.field(static=True)

    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_geom_attn: bool = False,
        use_plain_attn: bool = True,
        use_flash_attn: bool = False,
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
                key=key,
                inference=inference,
                dtype=dtype,
            )

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
        key, ffn_key = jax.random.split(key)
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(
                d_model,
                expansion_ratio,
                bias,
                key=key,
                dtype=dtype,
            )
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(
                d_model,
                expansion_ratio,
                bias,
                key=key,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.scaling_factor = residue_scaling_factor

    def __call__(
        self,
        x: Array,
        sequence_id: Array,
        frames: Affine3D,
        frames_mask: Array,
        chain_id: Array,
    ) -> Array:
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
        self.norm = eqx.nn.LayerNorm(d_model, use_bias=bias, dtype=dtype)


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
    sequence_head: eqx.nn.Sequential

    inference: bool

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        # tokenizer: EsmSequenceTokenizer,
        # use_flash_attn: bool = True,
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
