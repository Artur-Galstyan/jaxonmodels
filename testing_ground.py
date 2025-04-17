import equinox as eqx
import jax

mlp1 = eqx.nn.MLP(8, 8, 8, 8, key=jax.random.key(44))
mlp2 = eqx.nn.MLP(8, 8, 8, 8, key=jax.random.key(42))

o1 = mlp1(jax.numpy.ones(shape=(8)))
o2 = mlp2(jax.numpy.ones(shape=(8)))

print(jax.numpy.allclose(o1, o2))

mlp2 = eqx.tree_at(lambda x: x, mlp2, mlp1)


o2 = mlp2(jax.numpy.ones(shape=(8)))

print(jax.numpy.allclose(o1, o2))
