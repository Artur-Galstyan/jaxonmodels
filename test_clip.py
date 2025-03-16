import equinox as eqx
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
from jaxonmodels.models.clip import CLIP


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


# embed_dim = 512
# image_resolution = 224
# vision_layers = 12
# vision_width = 768
# vision_patch_size = 32
# context_length = 77
# vocab_size = 49408
# transformer_width = 512
# transformer_heads = 8
# transformer_layers = 12

embed_dim = 1024
image_resolution = 224
vision_layers = (3, 4, 6, 3)
vision_width = 64
vision_patch_size = None
context_length = 77
vocab_size = 49408
transformer_width = 512
transformer_heads = 8
transformer_layers = 12


clip, state = eqx.nn.make_with_state(CLIP)(
    embed_dim,
    image_resolution,
    vision_layers,
    vision_width,
    vision_patch_size,
    context_length,
    vocab_size,
    transformer_width,
    transformer_heads,
    transformer_layers,
)

text = F.clip_tokenize(["a photo of a human", "a photo of a cat", "a photo of a dog"])
# text = tokenize(["a photo of a cat"])
# print(text)
transform = _transform(image_resolution)
image = transform(Image.open("cat.jpg"))  # pyright: ignore

logits_per_image, logits_per_text, state = eqx.filter_vmap(
    clip, in_axes=(None, 0, None), out_axes=(0, 0, None), axis_name="batch"
)(jnp.array(image), text, state)
print(f"{logits_per_image.shape}=")
print(f"{logits_per_text.shape}=")

# image = transform(Image.open("cat.jpg")).unsqueeze(0)  # pyright: ignore

# output, state = eqx.filter_vmap(
#     clip, in_axes=(0, None, None), out_axes=(0, None), axis_name="batch"
# )(jnp.array(image), text, state)
