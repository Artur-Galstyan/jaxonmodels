import jax
import jax.numpy as jnp
from jaxonmodels.rnns.rnn import LSTM


n_letters = 26
n_categories = 26
learning_rate = 3e-4

key = jax.random.PRNGKey(33)
key, subkey = jax.random.split(key)

model = LSTM(input_size=n_letters, hidden_size=n_categories, key=subkey)

x = jnp.zeros(shape=(1, 26)).at[0].set(1)
print(x.shape)
out, hidden = model(x)

print(out.shape, hidden.shape)
print(out, hidden)
