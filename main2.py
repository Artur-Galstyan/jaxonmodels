import equinox as eqx
import jax
import jax.numpy as jnp
from jaxonmodels.vision.resnet import BasicBlock, ResNet


inplanes = 64
planes = 64
stride = 1
downsample = None
groups = 1
dilation = 1
norm_layer = eqx.nn.BatchNorm

key = jax.random.PRNGKey(22)

basic_block, state = eqx.nn.make_with_state(BasicBlock)(
    inplanes=inplanes,
    planes=planes,
    stride=stride,
    downsample=downsample,
    groups=groups,
    dilation=dilation,
    norm_layer=norm_layer,
    key=key,
)
x = jnp.ones(shape=(32, 64, 56, 56))
out, state = eqx.filter_vmap(
    basic_block, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
)(x, state)

net = ResNet(basic_block, layers=[1, 2, 3], key=jax.random.PRNGKey(22))
