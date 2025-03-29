import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree


def kaiming_init_conv2d(model: PyTree, state: eqx.nn.State, key: PRNGKeyArray):
    """Applies Kaiming He normal initialization to Conv2d weights."""
    # Filter function to identify Conv2d layers
    is_conv2d = lambda x: isinstance(x, eqx.nn.Conv2d)

    # Function to get weights (leaves) based on the filter
    def get_weights(model):
        return [
            x.weight for x in jax.tree.leaves(model, is_leaf=is_conv2d) if is_conv2d(x)
        ]

    # Get the list of current weights
    weights = get_weights(model)
    if not weights:  # If no Conv2d layers found
        return model, state

    # Create new weights using He initializer
    initializer = jax.nn.initializers.he_normal()
    # Split key for each weight tensor
    subkeys = jax.random.split(key, len(weights))
    new_weights = [
        initializer(subkeys[i], w.shape, w.dtype)  # Use original weight's dtype
        for i, w in enumerate(weights)
    ]

    # Replace old weights with new weights in the model pytree
    model = eqx.tree_at(get_weights, model, new_weights)

    return model, state
