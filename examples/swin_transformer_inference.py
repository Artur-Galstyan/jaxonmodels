import functools
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp

# Keep torch only for transforms and potentially labels
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import Swin_T_Weights, Swin_V2_T_Weights

from jaxonmodels.models.swin_transformer import load_swin_transformer


def load_imagenet_labels(pytorch_weights_enum):
    """Load ImageNet class labels from the PyTorch WeightsEnum metadata."""
    weights = pytorch_weights_enum.DEFAULT
    if not hasattr(weights, "meta") or "categories" not in weights.meta:
        # Fallback or error if categories aren't available
        print(
            f"Warning: Could not load categories from {pytorch_weights_enum}. Returning generic labels."  # noqa
        )
        return [f"Class_{i}" for i in range(1000)]  # Example fallback
    return weights.meta["categories"]


def main():
    image_path = "cat.jpg"
    dtype = jnp.float32

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # --- Data structures to hold results ---
    all_results = {"image": image_path, "models": []}

    # --- 1. Process Swin Transformer V1 Tiny ---
    print("-" * 30)
    print("Processing: JAX Swin Transformer V1 Tiny")
    model_name_v1 = "swin_t"
    weights_name_v1 = "swin_t_IMAGENET1K_V1"

    # Load JAX Model
    model_v1, state_v1 = load_swin_transformer(
        model_name=model_name_v1, weights=weights_name_v1, inference=True, dtype=dtype
    )
    model_v1, state_v1 = eqx.nn.inference_mode((model_v1, state_v1))

    # Define Preprocessing Manually for V1
    preprocess_v1 = transforms.Compose(
        [
            transforms.Resize(232, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("Using V1 preprocessing: Resize(232), CenterCrop(224)")

    # Load Image, Preprocess, Convert to JAX
    img_v1 = Image.open(image_path).convert("RGB")
    input_tensor_v1 = preprocess_v1(img_v1)
    input_batch_v1 = jnp.array(input_tensor_v1.unsqueeze(0).numpy(), dtype=dtype)  # pyright: ignore

    # Run Inference V1
    model_v1_pt = functools.partial(model_v1, key=jax.random.key(22))
    output_logits_v1, state_v1 = eqx.filter_vmap(
        model_v1_pt, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(input_batch_v1, state_v1)

    # Get Top 5 V1
    probabilities_v1 = jax.nn.softmax(output_logits_v1[0], axis=-1)
    top_indices_v1 = jnp.argsort(probabilities_v1)[::-1][:5]
    top_probs_v1 = probabilities_v1[top_indices_v1]
    labels_v1 = load_imagenet_labels(Swin_T_Weights)  # Get labels using V1 weights meta
    top_categories_v1 = [labels_v1[idx] for idx in top_indices_v1]

    # Store V1 Results
    results_v1 = {
        "model": f"JAX {model_name_v1}",
        "predictions": [
            {"category": cat, "probability": float(prob)}  # Convert JAX array to float
            for cat, prob in zip(top_categories_v1, top_probs_v1)
        ],
    }
    all_results["models"].append(results_v1)
    print("V1 Inference Complete.")

    # --- 2. Process Swin Transformer V2 Tiny ---
    print("-" * 30)
    print("Processing: JAX Swin Transformer V2 Tiny")
    model_name_v2 = "swin_v2_t"
    weights_name_v2 = "swin_v2_t_IMAGENET1K_V1"

    # Load JAX Model
    model_v2, state_v2 = load_swin_transformer(
        model_name=model_name_v2, weights=weights_name_v2, inference=True, dtype=dtype
    )
    model_v2, state_v2 = eqx.nn.inference_mode((model_v2, state_v2))

    # Define Preprocessing Manually for V2
    preprocess_v2 = transforms.Compose(
        [
            transforms.Resize(260, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("Using V2 preprocessing: Resize(260), CenterCrop(256)")

    # Load Image, Preprocess, Convert to JAX
    img_v2 = Image.open(image_path).convert(
        "RGB"
    )  # Reload image (or reuse if transforms allow)
    input_tensor_v2 = preprocess_v2(img_v2)
    input_batch_v2 = jnp.array(input_tensor_v2.unsqueeze(0).numpy(), dtype=dtype)  # pyright: ignore

    # Run Inference V2 (reuse JITed functions)

    model_v2_pt = functools.partial(model_v2, key=jax.random.key(22))
    output_logits_v2, state_v2 = eqx.filter_vmap(
        model_v2_pt, in_axes=(0, None), out_axes=(0, None), axis_name="batch"
    )(input_batch_v2, state_v2)

    # Get Top 5 V2
    probabilities_v2 = jax.nn.softmax(output_logits_v2[0], axis=-1)
    top_indices_v2 = jnp.argsort(probabilities_v2)[::-1][:5]
    top_probs_v2 = probabilities_v2[top_indices_v2]
    labels_v2 = load_imagenet_labels(
        Swin_V2_T_Weights
    )  # Get labels using V2 weights meta
    top_categories_v2 = [labels_v2[idx] for idx in top_indices_v2]

    # Store V2 Results
    results_v2 = {
        "model": f"JAX {model_name_v2}",
        "predictions": [
            {"category": cat, "probability": float(prob)}
            for cat, prob in zip(top_categories_v2, top_probs_v2)
        ],
    }
    all_results["models"].append(results_v2)
    print("V2 Inference Complete.")

    # --- 3. Print Combined Results ---
    print("-" * 30)
    print(f"\n--- Combined JAX Inference Results ({image_path}) ---")
    for model_result in all_results["models"]:
        print(f"\n{model_result['model']} Results:")
        for i, pred in enumerate(model_result["predictions"]):
            print(f"{i + 1}. {pred['category']}: {pred['probability']:.4f}")

    # --- 4. Save Combined Results to JSON ---
    output_filename = "swin_transformer_jax_results.json"
    try:
        with open(output_filename, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nCombined JAX Results saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving results to JSON: {e}")


if __name__ == "__main__":
    # Ensure you have 'swin_jax.py' and 'cat.jpg' accessible
    main()
