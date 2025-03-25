import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def main():
    # Load EfficientNet with pre-trained weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # If you want to load without pre-trained weights, use this instead:
    # model = efficientnet_b0(weights=None)

    # Set the model to evaluation mode
    model.eval()

    # Load and preprocess the image
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

    # Move the input and model to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model = model.to("cuda")

    print(f"Running inference on {img_path}...")

    with torch.no_grad():
        output = model(input_batch)

    # Load ImageNet class labels
    imagenet_labels = load_imagenet_labels()

    # Get the top 5 predictions
    _, indices = torch.topk(output, 5)
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    print("\nTop 5 predictions:")
    for i, idx in enumerate(indices[0]):
        idx = idx.item()
        print(
            f"{i + 1}. {imagenet_labels[idx]} ({probabilities[idx].item() * 100:.2f}%)"
        )

    # Print the dtypes of the model weights
    print("\nModel weights dtypes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")


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
