import equinox as eqx
import jax
import jax.numpy as jnp


def kaiming_init_conv2d(model, state, key):
    """Initialize model with Kaiming initialization"""
    key, subkey = jax.random.split(key)

    initializer = jax.nn.initializers.he_normal()
    is_conv2d = lambda x: isinstance(x, eqx.nn.Conv2d)
    get_weights = lambda m: [
        x.weight for x in jax.tree.leaves(m, is_leaf=is_conv2d) if is_conv2d(x)
    ]
    weights = get_weights(model)
    new_weights = [
        initializer(subkey, weight.shape, jnp.float32)
        for weight, subkey in zip(weights, jax.random.split(subkey, len(weights)))
    ]
    model = eqx.tree_at(get_weights, model, new_weights)

    return model, state
