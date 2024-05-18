import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxonmodels.rnns.rnn import RNN
from jaxtyping import Array, Float, PyTree


n_letters = 26
hidden_size = 32
n_categories = 26
learning_rate = 3e-4

key = jax.random.PRNGKey(33)
key, subkey = jax.random.split(key)

model = RNN(
    input_size=n_letters, hidden_size=hidden_size, output_size=n_categories, key=subkey
)

data = [jnp.eye(26)[i] for i in range(26)]  # 26 one-hot vectors
targets = jnp.eye(26)[
    jnp.array([(i + 1) % 26 for i in range(26)])
]  # One-hot encoded targets


def loss_fn(
    model: PyTree,
    x: Float[Array, " input_size"],
    y: Float[Array, " output_size"],
    hidden: Float[Array, " hidden_size"],
) -> tuple[Array, Array]:
    output, hidden = model(x, hidden)
    return optax.losses.softmax_cross_entropy(output, y).mean(), hidden


@eqx.filter_jit
def step(
    model: PyTree,
    x: Float[Array, " input_size"],
    y: Float[Array, " output_size"],
    hidden: Float[Array, " hidden_size"],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[PyTree, Array, Array]:
    ((loss, hidden), grads) = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, x, y, hidden
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, loss, hidden


optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

for epoch in range(1000):  # Adjust number of epochs as needed
    loss = 0
    hidden = model.init_hidden()

    for i in range(len(data)):
        input = data[i].reshape(1, -1)
        target = jnp.array([targets[i]])
        model, loss, hidden = step(model, input, target, hidden, optimizer, opt_state)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss: {loss}")

print("Training complete.")


def generate_sequence(start_letter, length):
    input = jnp.eye(26)[ord(start_letter) - ord("A")].reshape(
        1, -1
    )  # One-hot encode start letter
    hidden = model.init_hidden()
    sequence = start_letter

    for _ in range(length):
        output, hidden = model(input, hidden)
        predicted_index = int(jnp.argmax(output))
        next_letter = chr(predicted_index + ord("A"))
        sequence += next_letter
        input = jnp.eye(26)[predicted_index].reshape(1, -1)

    return sequence


generated_sequence = generate_sequence("A", 10)
print(f"Generated sequence: {generated_sequence}")
