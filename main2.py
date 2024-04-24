import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxonmodels.vision.resnet import resnet18 as jax_resnet18
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


key = jax.random.PRNGKey(22)


def test_on_image(img_name: str, torch_model, jax_model, state):
    # Load a pre-trained ResNet
    torch_model.eval()

    # Prepare an image
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(img_name)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)  # pyright:ignore

    # Predict
    with torch.no_grad():
        output = torch_model(batch_t)
        print(output.shape)
        _, predicted = torch.max(output, 1)

    print(
        f"Predicted: {predicted.item()}"
    )  # Outputs the ImageNet class index of the prediction
    np.save("torch.resnet", output.numpy())
    import json
    import urllib.request

    # Load ImageNet labels
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    with urllib.request.urlopen(url) as url:
        imagenet_labels = json.loads(url.read().decode())

    label = imagenet_labels[str(predicted.item())][1]
    print(f"Label for index {predicted.item()}: {label}")

    img_t = jnp.array(img_t.numpy())  # pyright:ignore
    img_t = jnp.expand_dims(img_t, axis=0)
    img_t.shape

    out, state = eqx.filter_vmap(
        jax_model, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(img_t, state)

    output = jnp.argmax(out)
    label = imagenet_labels[str(output)][
        1
    ]  # The '1' index contains the human-readable label
    print(f"Label for index {output}: {label}")
    np.save("jax.resnet", np.array(out))
    return jax_model, state


torch_model = resnet18(weights=ResNet18_Weights.DEFAULT)
jax_model, state = jax_resnet18(load_weights=True)
imgs = [
    "doggo.jpeg",
]  # "bucket.jpeg", "testimg.jpeg"]

for img in imgs:
    jax_model, state = test_on_image(img, torch_model, jax_model, state)
