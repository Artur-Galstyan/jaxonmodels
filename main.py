import json

import jax
import jax.numpy as jnp
import numpy as np
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import equinox as eqx
from jaxonmodels.models.alexnet import AlexNet

a = eqx.filter_jit(AlexNet.with_weights())

alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
alexnet.eval()


response = requests.get(
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)
idx_to_label = json.loads(response.content)
labels = [idx_to_label[str(i)][1] for i in range(1000)]

image_path = "cat.jpeg"  # Replace with your image path
image = Image.open(image_path)

preprocess = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # pyright: ignore
with torch.no_grad():
    output = alexnet(input_batch)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_indices = torch.topk(probabilities, 5)

print("Top 5 predictions:")
for i in range(5):
    print(f"{labels[top5_indices[i]]}: {top5_prob[i].item() * 100:.2f}%")

# Check if cat-related classes are in top predictions
cat_classes = ["tabby", "tiger_cat", "persian_cat", "siamese_cat", "egyptian_cat"]
top_prediction = labels[top5_indices[0]]
if any(cat in top_prediction.lower() for cat in cat_classes):
    print("\nSuccess! The model correctly identified a cat.")


o = a(jnp.array(np.array(input_tensor)), inference=True)
probs = jax.nn.softmax(o)
top5_prob, top5_indices = jax.lax.top_k(probs, 5)

print("Top 5 predictions:")
for i in range(5):
    print(f"{labels[top5_indices[i]]}: {top5_prob[i].item() * 100:.2f}%")


# Check if cat-related classes are in top predictions
cat_classes = ["tabby", "tiger_cat", "persian_cat", "siamese_cat", "egyptian_cat"]
top_prediction = labels[top5_indices[0]]
if any(cat in top_prediction.lower() for cat in cat_classes):
    print("\nSuccess! The model correctly identified a cat.")
