import jax


def fn(x, key):
    return jax.random.normal(key=key) + x


n = 5
key = jax.random.key(1)
keys = jax.random.split(key, n)
o1 = jax.vmap(fn, in_axes=(0, None))(jax.numpy.arange(n), key)
print(o1)

o2 = jax.vmap(fn)(jax.numpy.arange(n), keys)
print(o2)
