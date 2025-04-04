import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import models, transforms

# Path to the image
image_path = "cat.jpg"


def process_image_with_convnext():
    # 1. Load the image
    try:
        img = Image.open(image_path)
        print(f"Successfully loaded image: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Display the original image (optional)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title("Original Image")
    plt.savefig("original_cat.png")  # Save for viewing if not in interactive mode

    # 3. Preprocess the image for the model
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # 4. Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)
    print(f"Using device: {device}")

    # 5. Load the ConvNeXt model from torchvision
    try:
        # Load ConvNeXt model - you can choose from base, large, or small
        model = models.convnext_base(pretrained=True)
        model = model.to(device)
        model.eval()
        print("Successfully loaded ConvNeXt model")
    except Exception as e:
        print(f"Error loading ConvNeXt model: {e}")
        # For newer PyTorch versions, try the newer syntax if the above fails
        try:
            model = models.convnext_base(weights="IMAGENET1K_V1")
            model = model.to(device)
            model.eval()
            print("Successfully loaded ConvNeXt model with newer weights syntax")
        except Exception as e2:
            print(f"Error with alternative loading method: {e2}")
            return

    # 6. Run inference
    with torch.no_grad():
        try:
            output = model(input_batch)
            print("Successfully ran inference with ConvNeXt")
        except Exception as e:
            print(f"Error during inference: {e}")
            return

    # 7. Process the output - ConvNeXt is trained on ImageNet, so we get class predictions
    try:
        # Get the predicted class
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)

        # Load ImageNet class labels (you may need to download this)
        try:
            with open("imagenet_classes.txt") as f:
                classes = [line.strip() for line in f.readlines()]
            top5_classes = [classes[idx] for idx in top5_indices]
            print("Top 5 predictions:")
            for i in range(5):
                print(f"{top5_classes[i]}: {top5_prob[i].item():.4f}")
        except:
            # If class names file is not available
            print("Top 5 class indices:", top5_indices.cpu().numpy())
            print("Top 5 probabilities:", top5_prob.cpu().numpy())

    except Exception as e:
        print(f"Error processing output: {e}")
        return

    print("Processing complete")


if __name__ == "__main__":
    process_image_with_convnext()
