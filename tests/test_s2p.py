import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array, Float

from jaxonmodels.layers import BatchNorm
from jaxonmodels.statedict2pytree.s2p import (
    convert,
    move_running_fields_to_the_end,
    pytree_to_fields,
    state_dict_to_fields,
)


def test_conversion():
    class JaxModel(eqx.Module):
        conv1: eqx.nn.Conv1d
        norm1: BatchNorm
        conv2: eqx.nn.Conv1d
        pool: eqx.nn.MaxPool1d
        linear1: eqx.nn.Linear
        linear2: eqx.nn.Linear

        def __init__(self):
            # First conv layer: (3, 8) -> (6, 8)
            self.conv1 = eqx.nn.Conv1d(
                in_channels=3,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                key=jax.random.key(0),
            )
            self.norm1 = BatchNorm(6, axis_name="batch")

            # Second conv layer: (6, 8) -> (12, 8)
            self.conv2 = eqx.nn.Conv1d(
                in_channels=6,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=1,
                key=jax.random.key(2),
            )

            # Max pooling: (12, 8) -> (12, 4)
            self.pool = eqx.nn.MaxPool1d(kernel_size=2, stride=2)

            # After pooling: (12, 4) -> Flatten: 12*4 = 48
            self.linear1 = eqx.nn.Linear(
                in_features=48, out_features=24, key=jax.random.key(3)
            )

            self.linear2 = eqx.nn.Linear(
                in_features=24, out_features=16, key=jax.random.key(4)
            )

        def __call__(
            self, x: Float[Array, "3 8"], state: eqx.nn.State, inference: bool = False
        ):
            # First conv block
            x = self.conv1(x)
            x, state = self.norm1(x, state, inference=inference)
            x = jax.nn.relu(x)

            # Second conv block
            x = self.conv2(x)
            x = jax.nn.relu(x)

            # Pooling
            x = self.pool(x)

            # Flatten
            x = jnp.reshape(x, -1)

            # First dense block
            x = self.linear1(x)
            x = jax.nn.relu(x)

            # Output layer
            return self.linear2(x), state

    class TorchModel(torch.nn.Module):
        def __init__(self):
            super(TorchModel, self).__init__()

            # First conv block
            self.conv1 = torch.nn.Conv1d(
                in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
            )
            self.norm1 = torch.nn.BatchNorm1d(6)

            # Second conv block
            self.conv2 = torch.nn.Conv1d(
                in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1
            )

            # Pooling
            self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)

            # Linear layers
            self.linear1 = torch.nn.Linear(in_features=48, out_features=24)
            self.linear2 = torch.nn.Linear(in_features=24, out_features=16)

            # Activation and regularization
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            # First conv block
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)

            # Second conv block
            x = self.conv2(x)
            x = self.relu(x)

            # Pooling
            x = self.pool(x)

            # Flatten each example in the batch individually
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

            # First dense block
            x = self.linear1(x)
            x = self.relu(x)

            # Output layer
            return self.linear2(x)

    torch_model = TorchModel()
    with torch.no_grad():
        for param in torch_model.parameters():
            param.fill_(2.0)

    state_dict = torch_model.state_dict()

    jax_model, state = eqx.nn.make_with_state(JaxModel)()
    jax_model, state = convert(
        state_dict,
        (jax_model, state),
        *pytree_to_fields((jax_model, state)),
        move_running_fields_to_the_end(state_dict_to_fields(state_dict)),
    )

    np.random.seed(42)
    test_input = np.random.randn(2, 3, 8)

    torch_model.eval()
    torch_output = torch_model.forward(torch.Tensor(test_input))

    jax_output, state = eqx.filter_vmap(
        ft.partial(jax_model, inference=True),
        in_axes=(0, None),
        out_axes=(0, None),
        axis_name="batch",
    )(jnp.array(test_input), state)

    assert np.allclose(np.array(jax_output), torch_output.detach().numpy(), atol=1e-5)

    test_input = np.random.randn(2, 3, 8)

    torch_model.train()
    torch_output = torch_model.forward(torch.Tensor(test_input))

    jax_output, state = eqx.filter_vmap(
        ft.partial(jax_model, inference=False),
        in_axes=(0, None),
        out_axes=(0, None),
        axis_name="batch",
    )(jnp.array(test_input), state)

    assert np.allclose(np.array(jax_output), torch_output.detach().numpy(), atol=1e-5)

    jax_output, state = eqx.filter_vmap(
        ft.partial(jax_model, inference=False),
        in_axes=(0, None),
        out_axes=(0, None),
        axis_name="batch",
    )(jnp.array(test_input), state)

    assert np.allclose(np.array(jax_output), torch_output.detach().numpy(), atol=1e-5)
