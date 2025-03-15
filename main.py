import clip
import torch
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)

# Prepare your text and image
text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)

# Get features
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Normalize features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

# Calculate similarity
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(f"Similarity: {similarity}")

# If you want to compare with your model's outputs:
print(
    f"Image embedding shape: {image_features.shape}"
)  # Should be [1, 512] for ViT-B/32
print(
    f"Text embedding shape: {text_features.shape}"
)  # Should be [2, 512] for 2 text prompts
