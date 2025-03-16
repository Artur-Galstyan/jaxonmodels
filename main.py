import clip
import torch
import torch.nn.functional as F
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("RN50", device=device)

# Prepare your text and image
text = clip.tokenize(["a photo of a human", "a photo of a cat", "a photo of a dog"]).to(
    device
)
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)

# Get features and compute probabilities
with torch.no_grad():
    image_features_logits, text_features_logits = model.forward(image, text)
    print(f"{image_features_logits.shape}=")
    print(f"{text_features_logits.shape}=")

    # Now the probabilities will be meaningful
    image_probs = F.softmax(image_features_logits, dim=1)

    # Get probabilities for each text option
    probs = image_probs.cpu().numpy()[0]
    options = ["human", "cat", "dog"]

    for option, prob in zip(options, probs):
        print(f"Probability that the image is a {option}: {prob * 100:.2f}%")
