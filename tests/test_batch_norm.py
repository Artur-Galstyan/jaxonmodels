import equinox as eqx
import jax.numpy as jnp
import numpy as np
import torch
from jaxonlayers.layers import BatchNorm
from tqdm import tqdm


def test_batch_norm_against_pytorch_ndim_1():
    n_batches = 512
    size = 64
    batch_size = 128
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    inference: bool = False
    bn, bs = eqx.nn.make_with_state(BatchNorm)(
        size=size,
        axis_name="batch",
        eps=eps,
        momentum=momentum,
        affine=affine,
        inference=inference,
    )
    tbn = torch.nn.BatchNorm1d(
        num_features=size,
        eps=1e-5,
        momentum=0.1,
        affine=True,
    )
    np.random.seed(42)
    for b in tqdm(range(n_batches)):
        x = np.random.randn(batch_size, size)
        x = np.array(x, dtype=np.float32)
        x_t = torch.from_numpy(x)
        x_j = jnp.array(x)

        o_j, bs = eqx.filter_vmap(
            bn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )(x_j, bs)

        o_t = tbn(x_t)

        assert np.allclose(np.array(o_j), o_t.detach().numpy(), atol=1e-6), (
            f"Failed at batch {b}"
        )

        running_mean_j = bs.get(bn.state_index)[0]
        running_mean_t = tbn.running_mean
        assert running_mean_t is not None
        assert np.allclose(
            np.array(running_mean_j), running_mean_t.numpy(), atol=1e-6
        ), f"Failed at batch {b}"

        running_var_j = bs.get(bn.state_index)[1]
        running_var_t = tbn.running_var

        assert running_var_t is not None
        assert np.allclose(np.array(running_var_j), running_var_t.numpy(), atol=1e-6), (
            f"Failed at batch {b}"
        )


def test_batch_norm_against_pytorch_ndim_2():
    n_batches = 256
    size = 16
    n_dims = 3
    batch_size = 32
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    inference: bool = False
    bn, bs = eqx.nn.make_with_state(BatchNorm)(
        size=size,
        axis_name="batch",
        eps=eps,
        momentum=momentum,
        affine=affine,
        inference=inference,
    )
    tbn = torch.nn.BatchNorm2d(
        num_features=size,
        eps=1e-5,
        momentum=0.1,
        affine=True,
    )
    np.random.seed(42)
    for b in tqdm(range(n_batches)):
        x = np.random.randn(*((batch_size,) + (size,) * n_dims))
        x = np.array(x, dtype=np.float32)
        x_t = torch.from_numpy(x)
        x_j = jnp.array(x)

        o_j, bs = eqx.filter_vmap(
            bn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )(x_j, bs)

        o_t = tbn(x_t)

        assert np.allclose(np.array(o_j), o_t.detach().numpy(), atol=1e-6), (
            f"Failed at batch {b}"
        )

        running_mean_j = bs.get(bn.state_index)[0]
        running_mean_t = tbn.running_mean
        assert running_mean_t is not None
        assert np.allclose(
            np.array(running_mean_j), running_mean_t.numpy(), atol=1e-6
        ), f"Failed at batch {b}"

        running_var_j = bs.get(bn.state_index)[1]
        running_var_t = tbn.running_var

        assert running_var_t is not None
        assert np.allclose(np.array(running_var_j), running_var_t.numpy(), atol=1e-6), (
            f"Failed at batch {b}"
        )


def test_batch_norm_against_pytorch_ndim_3():
    n_batches = 128
    size = 8
    n_dims = 4
    batch_size = 16
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True
    inference: bool = False
    bn, bs = eqx.nn.make_with_state(BatchNorm)(
        size=size,
        axis_name="batch",
        eps=eps,
        momentum=momentum,
        affine=affine,
        inference=inference,
    )
    tbn = torch.nn.BatchNorm3d(
        num_features=size,
        eps=1e-5,
        momentum=0.1,
        affine=True,
    )
    np.random.seed(42)
    for b in tqdm(range(n_batches)):
        x = np.random.randn(*((batch_size,) + (size,) * n_dims))
        x = np.array(x, dtype=np.float32)
        x_t = torch.from_numpy(x)
        x_j = jnp.array(x)

        o_j, bs = eqx.filter_vmap(
            bn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )(x_j, bs)

        o_t = tbn(x_t)

        assert np.allclose(np.array(o_j), o_t.detach().numpy(), atol=1e-6), (
            f"Failed at batch {b}"
        )

        running_mean_j = bs.get(bn.state_index)[0]
        running_mean_t = tbn.running_mean
        assert running_mean_t is not None
        assert np.allclose(
            np.array(running_mean_j), running_mean_t.numpy(), atol=1e-6
        ), f"Failed at batch {b}"

        running_var_j = bs.get(bn.state_index)[1]
        running_var_t = tbn.running_var

        assert running_var_t is not None
        assert np.allclose(np.array(running_var_j), running_var_t.numpy(), atol=1e-6), (
            f"Failed at batch {b}"
        )


def test_batch_norm_inference_ndim_1():
    n_batches_train = 64
    n_batches_infer = 32
    size = 64
    batch_size = 128
    eps: float = 1e-5
    momentum: float = 0.1
    affine: bool = True

    bn, bs = eqx.nn.make_with_state(BatchNorm)(
        size=size,
        axis_name="batch",
        eps=eps,
        momentum=momentum,
        affine=affine,
        inference=False,
    )

    tbn = torch.nn.BatchNorm1d(
        num_features=size,
        eps=1e-5,
        momentum=0.1,
        affine=True,
    )

    np.random.seed(42)
    for b in tqdm(range(n_batches_train), desc="Training BatchNorm1D"):
        x = np.random.randn(batch_size, size)
        x = np.array(x, dtype=np.float32)
        x_t = torch.from_numpy(x)
        x_j = jnp.array(x)

        # Forward passes
        o_j, bs = eqx.filter_vmap(
            bn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )(x_j, bs)
        o_t = tbn(x_t)

    running_mean_j = bs.get(bn.state_index)[0]
    running_mean_t = tbn.running_mean
    assert running_mean_t is not None
    assert np.allclose(np.array(running_mean_j), running_mean_t.numpy(), atol=1e-6), (
        "Running mean mismatch after training"
    )

    running_var_j = bs.get(bn.state_index)[1]
    running_var_t = tbn.running_var
    assert running_var_t is not None
    assert np.allclose(np.array(running_var_j), running_var_t.numpy(), atol=1e-6), (
        "Running var mismatch after training"
    )

    # Switch to inference mode
    bn = eqx.tree_at(lambda m: m.inference, bn, True)
    tbn.eval()  # Switch PyTorch model to eval mode

    # Inference phase - verify models behave identically
    np.random.seed(24)
    for b in tqdm(range(n_batches_infer), desc="Testing BatchNorm1D inference"):
        x = np.random.randn(batch_size, size)
        x = np.array(x, dtype=np.float32)
        x_t = torch.from_numpy(x)
        x_j = jnp.array(x)

        # Forward passes in inference mode
        o_j, bs = eqx.filter_vmap(
            bn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )(x_j, bs)

        with torch.no_grad():
            o_t = tbn(x_t)

        # Verify outputs match during inference
        assert np.allclose(np.array(o_j), o_t.detach().numpy(), atol=1e-6), (
            f"Output mismatch during inference at batch {b}"
        )

        # Verify running stats haven't changed during inference
        new_running_mean_j = bs.get(bn.state_index)[0]
        new_running_mean_t = tbn.running_mean

        assert np.allclose(new_running_mean_j, running_mean_j, atol=1e-6), (
            f"JAX running mean changed during inference at batch {b}"
        )
        assert new_running_mean_t is not None
        assert np.allclose(
            new_running_mean_t.numpy(), running_mean_t.numpy(), atol=1e-6
        ), f"PyTorch running mean changed during inference at batch {b}"

        new_running_var_j = bs.get(bn.state_index)[1]
        new_running_var_t = tbn.running_var

        assert np.allclose(new_running_var_j, running_var_j, atol=1e-6), (
            f"JAX running var changed during inference at batch {b}"
        )
        assert new_running_var_t is not None
        assert np.allclose(
            new_running_var_t.numpy(), running_var_t.numpy(), atol=1e-6
        ), f"PyTorch running var changed during inference at batch {b}"


def test_batch_norm_inference_ndim_2():
    """Test that BatchNorm behaves like PyTorch's BatchNorm2d in inference mode."""
    # Setup parameters
    n_batches_train = 48  # Train on this many batches to build statistics
    n_batches_infer = 24  # Test inference mode on this many batches
    size = 16  # Feature size (channels)
    spatial_size = 8  # Size of spatial dimensions
    batch_size = 32  # Number of samples per batch
    eps: float = 1e-5  # Small constant for numerical stability
    momentum: float = 0.1  # Running statistics momentum
    affine: bool = True  # Use affine parameters

    # Initialize models in training mode
    bn, bs = eqx.nn.make_with_state(BatchNorm)(
        size=size,
        axis_name="batch",
        eps=eps,
        momentum=momentum,
        affine=affine,
        inference=False,  # Start in training mode
    )

    tbn = torch.nn.BatchNorm2d(
        num_features=size,
        eps=1e-5,
        momentum=0.1,
        affine=True,
    )

    # Training phase to build up statistics
    np.random.seed(42)
    for b in tqdm(range(n_batches_train), desc="Training BatchNorm2D"):
        shape = (batch_size, size, spatial_size, spatial_size)
        x = np.random.randn(*shape)
        x = np.array(x, dtype=np.float32)
        x_t = torch.from_numpy(x)
        x_j = jnp.array(x)

        # Forward passes
        o_j, bs = eqx.filter_vmap(
            bn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )(x_j, bs)
        o_t = tbn(x_t)

    # Verify final training stats match
    running_mean_j = bs.get(bn.state_index)[0]
    running_mean_t = tbn.running_mean
    assert running_mean_t is not None
    assert np.allclose(np.array(running_mean_j), running_mean_t.numpy(), atol=1e-6), (
        "Running mean mismatch after training"
    )

    running_var_j = bs.get(bn.state_index)[1]
    running_var_t = tbn.running_var
    assert running_var_t is not None
    assert np.allclose(np.array(running_var_j), running_var_t.numpy(), atol=1e-6), (
        "Running var mismatch after training"
    )

    # Switch to inference mode
    bn = eqx.tree_at(lambda m: m.inference, bn, True)
    tbn.eval()  # Switch PyTorch model to eval mode

    # Inference phase - verify models behave identically
    np.random.seed(24)
    for b in tqdm(range(n_batches_infer), desc="Testing BatchNorm2D inference"):
        shape = (batch_size, size, spatial_size, spatial_size)
        x = np.random.randn(*shape)
        x = np.array(x, dtype=np.float32)
        x_t = torch.from_numpy(x)
        x_j = jnp.array(x)

        # Forward passes in inference mode
        o_j, bs = eqx.filter_vmap(
            bn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )(x_j, bs)

        with torch.no_grad():
            o_t = tbn(x_t)

        # Verify outputs match during inference
        assert np.allclose(np.array(o_j), o_t.detach().numpy(), atol=1e-6), (
            f"Output mismatch during inference at batch {b}"
        )

        # Verify running stats haven't changed during inference
        new_running_mean_j = bs.get(bn.state_index)[0]
        new_running_mean_t = tbn.running_mean

        assert np.allclose(new_running_mean_j, running_mean_j, atol=1e-6), (
            f"JAX running mean changed during inference at batch {b}"
        )
        assert new_running_mean_t is not None
        assert np.allclose(
            new_running_mean_t.numpy(), running_mean_t.numpy(), atol=1e-6
        ), f"PyTorch running mean changed during inference at batch {b}"

        new_running_var_j = bs.get(bn.state_index)[1]
        new_running_var_t = tbn.running_var

        assert np.allclose(new_running_var_j, running_var_j, atol=1e-6), (
            f"JAX running var changed during inference at batch {b}"
        )
        assert new_running_var_t is not None
        assert np.allclose(
            new_running_var_t.numpy(), running_var_t.numpy(), atol=1e-6
        ), f"PyTorch running var changed during inference at batch {b}"


def test_batch_norm_inference_ndim_3():
    """Test that BatchNorm behaves like PyTorch's BatchNorm3d in inference mode."""
    # Setup parameters
    n_batches_train = 32  # Train on this many batches to build statistics
    n_batches_infer = 16  # Test inference mode on this many batches
    size = 8  # Feature size (channels)
    spatial_size = 4  # Size of spatial dimensions
    batch_size = 16  # Number of samples per batch
    eps: float = 1e-5  # Small constant for numerical stability
    momentum: float = 0.1  # Running statistics momentum
    affine: bool = True  # Use affine parameters

    # Initialize models in training mode
    bn, bs = eqx.nn.make_with_state(BatchNorm)(
        size=size,
        axis_name="batch",
        eps=eps,
        momentum=momentum,
        affine=affine,
        inference=False,  # Start in training mode
    )

    tbn = torch.nn.BatchNorm3d(
        num_features=size,
        eps=1e-5,
        momentum=0.1,
        affine=True,
    )

    # Training phase to build up statistics
    np.random.seed(42)
    for b in tqdm(range(n_batches_train), desc="Training BatchNorm3D"):
        # For 3D: [batch_size, channels, depth, height, width]
        shape = (batch_size, size, spatial_size, spatial_size, spatial_size)
        x = np.random.randn(*shape)
        x = np.array(x, dtype=np.float32)
        x_t = torch.from_numpy(x)
        x_j = jnp.array(x)

        # Forward passes
        o_j, bs = eqx.filter_vmap(
            bn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )(x_j, bs)
        o_t = tbn(x_t)

    # Verify final training stats match
    running_mean_j = bs.get(bn.state_index)[0]
    running_mean_t = tbn.running_mean
    assert running_mean_t is not None
    assert np.allclose(np.array(running_mean_j), running_mean_t.numpy(), atol=1e-6), (
        "Running mean mismatch after training"
    )

    running_var_j = bs.get(bn.state_index)[1]
    running_var_t = tbn.running_var
    assert running_var_t is not None
    assert np.allclose(np.array(running_var_j), running_var_t.numpy(), atol=1e-6), (
        "Running var mismatch after training"
    )

    # Switch to inference mode
    bn = eqx.tree_at(lambda m: m.inference, bn, True)
    tbn.eval()  # Switch PyTorch model to eval mode

    # Inference phase - verify models behave identically
    np.random.seed(24)
    for b in tqdm(range(n_batches_infer), desc="Testing BatchNorm3D inference"):
        shape = (batch_size, size, spatial_size, spatial_size, spatial_size)
        x = np.random.randn(*shape)
        x = np.array(x, dtype=np.float32)
        x_t = torch.from_numpy(x)
        x_j = jnp.array(x)

        # Forward passes in inference mode
        o_j, bs = eqx.filter_vmap(
            bn, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
        )(x_j, bs)

        with torch.no_grad():
            o_t = tbn(x_t)

        # Verify outputs match during inference
        assert np.allclose(np.array(o_j), o_t.detach().numpy(), atol=1e-6), (
            f"Output mismatch during inference at batch {b}"
        )

        # Verify running stats haven't changed during inference
        new_running_mean_j = bs.get(bn.state_index)[0]
        new_running_mean_t = tbn.running_mean

        assert np.allclose(new_running_mean_j, running_mean_j, atol=1e-6), (
            f"JAX running mean changed during inference at batch {b}"
        )
        assert new_running_mean_t is not None
        assert np.allclose(
            new_running_mean_t.numpy(), running_mean_t.numpy(), atol=1e-6
        ), f"PyTorch running mean changed during inference at batch {b}"

        new_running_var_j = bs.get(bn.state_index)[1]
        new_running_var_t = tbn.running_var

        assert np.allclose(new_running_var_j, running_var_j, atol=1e-6), (
            f"JAX running var changed during inference at batch {b}"
        )
        assert new_running_var_t is not None
        assert np.allclose(
            new_running_var_t.numpy(), running_var_t.numpy(), atol=1e-6
        ), f"PyTorch running var changed during inference at batch {b}"
