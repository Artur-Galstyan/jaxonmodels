import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import jaxonmodels.functions as F
from jaxonmodels.models.clip import load_clip


def resize_with_bicubic(image, target_size):
    """Resize an image using bicubic interpolation."""
    return image.resize((target_size, target_size), resample=Image.BICUBIC)  # pyright: ignore


def center_crop(image, crop_size):
    """Perform a center crop on an image."""
    width, height = image.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))


def convert_image_to_rgb(image):
    """Convert image to RGB."""
    return image.convert("RGB")


def to_jax_tensor(image):
    """Convert PIL image to JAX tensor with values in [0, 1]."""
    np_array = np.array(image).astype(np.float32) / 255.0
    return jnp.array(np_array)


def normalize(tensor, mean, std):
    """Normalize a tensor with mean and std."""
    mean = jnp.array(mean).reshape(3, 1, 1)
    std = jnp.array(std).reshape(3, 1, 1)
    return (tensor - mean) / std


def transform_image(image, n_px):
    """Apply the CLIP image transformation pipeline."""
    # Resize with bicubic interpolation
    image = resize_with_bicubic(image, n_px)

    # Center crop
    image = center_crop(image, n_px)

    # Convert to RGB
    image = convert_image_to_rgb(image)

    # Convert to tensor with values in [0, 1]
    tensor = to_jax_tensor(image)

    # Rearrange from HWC to CHW format
    tensor = jnp.transpose(tensor, (2, 0, 1))

    # Normalize with CLIP's mean and std values
    tensor = normalize(
        tensor,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    return tensor


# Set image resolution
image_resolution = 224

# Tokenize text prompts
text = F.clip_tokenize(["a photo of a human", "a photo of a cat", "a photo of a dog"])

# Load and transform the image
image_pil = Image.open("cat.jpg")
image = transform_image(image_pil, image_resolution)

# Load CLIP model
clip, state = load_clip(model="ViT-B/32", with_weights=True)
clip_pt = functools.partial(clip, inference=True)

# Run the model with vmap for batching
logits_per_image, logits_per_text, state = eqx.filter_vmap(
    clip_pt, in_axes=(None, 0, None), out_axes=(0, 0, None), axis_name="batch"
)(image, text, state)

print(f"{logits_per_image.shape=}, {logits_per_image=}")
print(f"{logits_per_text.shape=}, {logits_per_text=}")

# Get probabilities using softmax
image_probs = jax.nn.softmax(logits_per_image)
probs = image_probs

# Display classification results
options = ["human", "cat", "dog"]
for option, prob in zip(options, probs):
    print(f"Probability that the image is a {option}: {prob * 100:.2f}%")
