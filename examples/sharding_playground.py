import equinox as eqx
import jax
import jax.sharding as js
from jaxtyping import Array, Float


class Model(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear

    def __init__(self):
        self.lin1 = eqx.nn.Linear(12, 4, key=jax.random.key(1))
        self.lin2 = eqx.nn.Linear(4, 1, key=jax.random.key(2))

    def __call__(self, x: Float[Array, "n_features"]) -> Float[Array, "n_labels"]:
        x = self.lin1(x)
        x = jax.nn.relu(x)
        x = self.lin2(x)
        return x


model = Model()

batch_size = 4
data = jax.random.uniform(key=jax.random.key(22), shape=(batch_size, 12))


mesh = jax.make_mesh((2,), axis_names=("batch",), axis_types=(js.AxisType.Auto,))

# data_sharded = jax.device_put(data, js.NamedSharding(mesh, js.PartitionSpec("batch")))
data_sharded = eqx.filter_shard(data, js.NamedSharding(mesh, js.PartitionSpec("batch")))

jax.debug.visualize_array_sharding(data_sharded)
