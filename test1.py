import equinox as eqx
import jax
import jax.numpy as jnp
from jaxonmodels.vision.vgg import vgg13


key = jax.random.PRNGKey(22)

vgg, state = vgg13(key)

ones_image = jnp.ones((1, 3, 224, 224))

output, state = eqx.filter_vmap(
    vgg,
    axis_name="batch",
    in_axes=(0, None, None, None),
    out_axes=(0, None),
)(ones_image, state, jax.random.PRNGKey(33), jax.random.PRNGKey(34))
print(output.shape)
