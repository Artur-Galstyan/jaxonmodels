import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

import jaxonmodels.functions as F
from jaxonmodels.models.clip import load_clip


def convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


image_resolution = 224


text = F.clip_tokenize(["a photo of a human", "a photo of a cat", "a photo of a dog"])
# text = F.clip_tokenize(["a photo of a cat"])
# print(text)
transform = _transform(image_resolution)
image = transform(Image.open("cat.jpg"))  # pyright: ignore

# Add batch dimension to text and image
# text = jnp.expand_dims(text, axis=0)
# image = jnp.expand_dims(jnp.array(image), axis=0)

print(f"{text.shape=}, {image.shape=}")

clip, state = load_clip(model="ViT-B/32", with_weights=True)

clip_pt = functools.partial(clip, inference=True)
logits_per_image, logits_per_text, state = eqx.filter_vmap(
    clip_pt, in_axes=(None, 0, None), out_axes=(0, 0, None), axis_name="batch"
)(jnp.array(image), text, state)
print(f"{logits_per_image.shape=}, {logits_per_image=}")
print(f"{logits_per_text.shape=}, {logits_per_text=}")

# image = transform(Image.open("cat.jpg")).unsqueeze(0)  # pyright: ignore

# output, state = eqx.filter_vmap(
#     clip, in_axes=(0, None, None), out_axes=(0, None), axis_name="batch"
# )(jnp.array(image), text, state)

image_probs = jax.nn.softmax(logits_per_image)

# Get probabilities for each text option
probs = image_probs
options = ["human", "cat", "dog"]

for option, prob in zip(options, probs):
    print(f"Probability that the image is a {option}: {prob * 100:.2f}%")
