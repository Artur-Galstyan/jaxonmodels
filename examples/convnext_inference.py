import equinox as eqx
import jax
import jax.numpy as jnp
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ConvNeXt_Base_Weights

from jaxonmodels.models.convnext import load_convnext


def main():
    # Load the ConvNeXt model with pre-trained weights
    # Note: ConvNeXt doesn't return state like EfficientNet
    model, state = load_convnext(
        "convnext_base", weights="convnext_base_IMAGENET1K_V1", dtype=jnp.float32
    )

    model, state = eqx.nn.inference_mode((model, state))

    # Path to image
    img_path = "cat.jpg"
    img = Image.open(img_path)

    # Apply the transformations consistent with ConvNeXt training
    # Using the same normalization as in the PyTorch example
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Set random key for model inference
    key = jax.random.key(43)

    # Preprocess the image
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    input_batch = jnp.array(input_batch.numpy(), dtype=jnp.float32)
    print(f"{input_batch.shape=}")

    # Since ConvNeXt doesn't have state in your implementation
    # We can call it directly without the filter_vmap that handles state
    key, subkey = jax.random.split(key)

    # Apply model to input batch
    # Use vmap to handle the batch dimension
    output, state = eqx.filter_vmap(
        model, in_axes=(0, None, None), out_axes=(0, None), axis_name="batch"
    )(input_batch, state, subkey)
    print(f"{output.shape=}")

    # Load ImageNet class labels
    imagenet_labels = load_imagenet_labels()

    # Get the top 5 predictions
    probabilities = jax.nn.softmax(output, axis=1)[0]
    top_indices = jnp.argsort(probabilities)[::-1][:5]

    print("\nTop 5 predictions:")
    for i, idx in enumerate(top_indices):
        print(f"{i + 1}. {imagenet_labels[idx]} ({probabilities[idx]})")


def load_imagenet_labels():
    """
    Load ImageNet class labels from the torchvision built-in classes
    """
    # Get the class mapping directly from the weights
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]
    # Return the list of categories
    return categories


if __name__ == "__main__":
    main()
