import functools as ft
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import requests
from PIL import Image
from torchvision import transforms

from jaxonmodels.models.resnet import resnet50


def get_imagenet_labels():
    """Fetch ImageNet class labels"""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        labels = [line.strip() for line in response.text.splitlines()]
        return labels
    except Exception as e:
        print(f"Error fetching ImageNet labels: {e}")
        # Fallback to a minimal set for testing
        return ["background"] + [f"class_{i}" for i in range(1, 1000)]


def preprocess_image(image_path):
    """Load and preprocess an image for ResNet inference"""
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0).numpy()  # pyright: ignore
    return jnp.array(input_tensor)


def main():
    # Load model with pre-trained weights
    # r, s = resnet18(weights="resnet18_IMAGENET1K_V1", key=jax.random.key(0))
    r, s = resnet50(weights="resnet50_IMAGENET1K_V2", key=jax.random.key(0))
    # Set model to inference mode for batch normalization
    r = ft.partial(r, inference=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "cat.jpg")
    jax_input = preprocess_image(image_path)

    # Run inference
    jax_output, _ = eqx.filter_vmap(
        r, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(jax_input, s)

    print(f"Output shape: {jax_output.shape}")

    # Get class probabilities
    jax_probs = jax.nn.softmax(jax_output[0])

    # Get top predictions
    labels = get_imagenet_labels()
    top_indices = jnp.argsort(jax_probs)[-5:][::-1]

    print("\nTop 5 predictions:")
    for i, idx in enumerate(top_indices):
        prob = float(jax_probs[idx])
        print(f"{i + 1}. {labels[int(idx)]}: {prob:.4f}")


if __name__ == "__main__":
    main()
