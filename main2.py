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

# basic_block, state = eqx.nn.make_with_state(BasicBlock)(
#     inplanes=inplanes,
#     planes=planes,
#     stride=stride,
#     downsample=downsample,
#     groups=groups,
#     dilation=dilation,
#     norm_layer=norm_layer,
#     key=key,
# )
x = jnp.ones(shape=(32, 3, 224, 224))
# out, state = eqx.filter_vmap(
#     basic_block, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
# )(x, state)

net, state = eqx.nn.make_with_state(ResNet)(
    BasicBlock, layers=[2, 2, 2, 2], key=jax.random.PRNGKey(22)
)

out, state = eqx.filter_vmap(
    net, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
)(x, state)

print(out.shape)
