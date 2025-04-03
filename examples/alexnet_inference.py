import os

import equinox as eqx
import jax
import jax.numpy as jnp
import requests
from PIL import Image
from torchvision import transforms

from jaxonmodels.models.alexnet import alexnet


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


current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "cat.jpg")
jax_input = preprocess_image(image_path)[0]


anet = alexnet(with_weights=True, dtype=jnp.float16)
anet = eqx.nn.inference_mode(anet)
print(anet.conv2.weight.dtype)
jax_output = anet(jnp.array(jax_input, dtype=jnp.float16))
jax_probs = jax.nn.softmax(jax_output)

# Get top predictions
labels = get_imagenet_labels()
top_indices = jnp.argsort(jax_probs)[-5:][::-1]

print("\nTop 5 predictions:")
for i, idx in enumerate(top_indices):
    prob = float(jax_probs[idx])
    print(f"{i + 1}. {labels[int(idx)]}: {prob:.4f}")
