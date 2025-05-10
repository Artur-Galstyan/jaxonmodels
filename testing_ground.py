import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor  # pyright: ignore

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

IMAGE_PATH = "cat.jpg"  # Your image file

image = Image.open(IMAGE_PATH)

candidate_labels = ["cat", "dog", "human"]
texts = [f"This is a photo of {label}." for label in candidate_labels]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)  # these are the probabilities
print(probs)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")

# print(type(model))

# print(model.vision_model.embeddings.config)
