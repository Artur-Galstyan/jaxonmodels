import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights

from jaxonmodels.models.efficientnet import _efficientnet, _efficientnet_conf


def main():
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_b0", width_mult=1.0, depth_mult=1.0
    )

    key = jax.random.key(42)
    model, state = _efficientnet(
        inverted_residual_setting, 0.2, last_channel, False, key
    )

    img_path = "cat.jpg"
    img = Image.open(img_path)

    # Apply the same transformations used during training
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    input_batch = jnp.array(input_batch.numpy())
    print(f"{input_batch.shape=}")
    key, subkey = jax.random.split(key)
    model_pt = functools.partial(model, inference=True, key=subkey)
    output, state = eqx.filter_vmap(
        model_pt, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(input_batch, state)

    print(output.shape)

    # # Load ImageNet class labels
    # imagenet_labels = load_imagenet_labels()

    # # Get the top 5 predictions
    # _, indices = torch.topk(output, 5)
    # probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    # print("\nTop 5 predictions:")
    # for i, idx in enumerate(indices[0]):
    #     idx = idx.item()
    #     print(
    #         f"{i + 1}. {imagenet_labels[idx]} ({probabilities[idx].item() * 100:.2f}%)" # noqa
    #     )


def load_imagenet_labels():
    """
    Load ImageNet class labels from the torchvision built-in classes
    """
    # Get the class mapping directly from the weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]

    # Return the list of categories
    return categories


if __name__ == "__main__":
    main()
