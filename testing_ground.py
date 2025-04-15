import json

import torch
from PIL import Image
from torchvision.models import Swin_T_Weights, Swin_V2_T_Weights, swin_t, swin_v2_t


def load_image(image_path):
    """Load and preprocess the input image."""
    image = Image.open(image_path).convert("RGB")
    return image


def inference_swin_v1(image_path):
    """Perform inference using Swin Transformer V1."""
    # Load pre-trained Swin Transformer V1 model
    weights = Swin_T_Weights.DEFAULT
    model = swin_t(weights=weights)
    model.eval()

    # Get the transformation pipeline from the weights
    preprocess = weights.transforms()

    # Load and preprocess the image
    image = load_image(image_path)
    input_tensor = preprocess(image)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Move the input to the same device as the model
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model = model.to("cuda")

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    # Convert to Python lists
    top5_prob = top5_prob.cpu().numpy().tolist()
    top5_indices = top5_indices.cpu().numpy().tolist()

    # Get class names
    categories = weights.meta["categories"]
    top5_categories = [categories[idx] for idx in top5_indices]

    # Create results dictionary
    results = {
        "model": "Swin Transformer V1",
        "predictions": [
            {"category": cat, "probability": prob}
            for cat, prob in zip(top5_categories, top5_prob)
        ],
    }

    return results


def inference_swin_v2(image_path):
    print("SWIN V2")
    """Perform inference using Swin Transformer V2."""
    # Load pre-trained Swin Transformer V2 model
    weights = Swin_V2_T_Weights.DEFAULT
    model = swin_v2_t(weights=weights)
    model.eval()

    # Get the transformation pipeline from the weights
    preprocess = weights.transforms()

    # Load and preprocess the image
    image = load_image(image_path)
    input_tensor = preprocess(image)

    # Add batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Move the input to the same device as the model
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model = model.to("cuda")

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    # Convert to Python lists
    top5_prob = top5_prob.cpu().numpy().tolist()
    top5_indices = top5_indices.cpu().numpy().tolist()

    # Get class names
    categories = weights.meta["categories"]
    top5_categories = [categories[idx] for idx in top5_indices]

    # Create results dictionary
    results = {
        "model": "Swin Transformer V2",
        "predictions": [
            {"category": cat, "probability": prob}
            for cat, prob in zip(top5_categories, top5_prob)
        ],
    }

    return results


def main():
    image_path = "cat.jpg"

    # Perform inference with both models
    v1_results = inference_swin_v1(image_path)
    v2_results = inference_swin_v2(image_path)

    # Print results
    print("\nSwin Transformer V1 Results:")
    for i, pred in enumerate(v1_results["predictions"]):
        print(f"{i + 1}. {pred['category']}: {pred['probability']:.4f}")

    print("\nSwin Transformer V2 Results:")
    for i, pred in enumerate(v2_results["predictions"]):
        print(f"{i + 1}. {pred['category']}: {pred['probability']:.4f}")

    # Save results to a JSON file
    combined_results = {"image": image_path, "models": [v1_results, v2_results]}

    with open("swin_transformer_results.json", "w") as f:
        json.dump(combined_results, f, indent=4)

    print("\nResults saved to swin_transformer_results.json")


if __name__ == "__main__":
    main()
