import json
import urllib.request

import torch
from jaxonmodels.vision.resnet import resnet18 as resnet18_jax
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


img_name = "doggo.jpeg"
resnet = resnet18(weights=None)

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
    output = resnet(batch_t)
    print(output.shape)
    _, predicted = torch.max(output, 1)

print(
    f"Predicted: {predicted.item()}"
)  # Outputs the ImageNet class index of the prediction

# Load ImageNet labels
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urllib.request.urlopen(url) as url:
    imagenet_labels = json.loads(url.read().decode())

label = imagenet_labels[str(predicted.item())][1]
print(f"Label for index {predicted.item()}: {label}")


jax_resnet = resnet18_jax()
