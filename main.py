import os  # To check if files exist

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# --- Configuration ---
IMAGE_PATH = "cat.jpg"  # Your image file
# Choose the checkpoint you downloaded
# CHECKPOINT_PATH = 'sam_vit_h_4b8939.pth'
# CHECKPOINT_PATH = 'sam_vit_l_0b3195.pth'
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"  # Using ViT-B as an example
MODEL_TYPE = "vit_b"  # Must match the checkpoint (vit_b, vit_l, or vit_h)
OUTPUT_IMAGE_PATH = "cat_segmented.png"  # Where to save the result

# --- Check if files exist ---
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file not found at '{IMAGE_PATH}'")
    exit()
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Model checkpoint file not found at '{CHECKPOINT_PATH}'")
    print("Please download the checkpoint file and place it here or update the path.")
    exit()


# --- Function to show masks (from SAM repo examples) ---
def show_anns(anns, image):
    if len(anns) == 0:
        return
    # Sort annotations by area to draw larger masks first
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Create a blank overlay image
    mask_overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)  # RGBA

    for ann in sorted_anns:
        m = ann["segmentation"]
        # Generate a random color (RGBA)
        color_mask = np.concatenate(
            [np.random.random(3) * 255, [150]]
        )  # RGB + Alpha (semi-transparent)
        # Apply color where the mask is True
        mask_overlay[m, :] = color_mask

    # Blend the overlay with the original image
    ax.imshow(image)
    ax.imshow(mask_overlay.astype(np.uint8))  # Display the colored masks


# --- Setup SAM ---
print("Setting up SAM model...")
# Use CUDA if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=device)

# Create the Automatic Mask Generator
mask_generator = SamAutomaticMaskGenerator(sam)
print("SAM model loaded.")

# --- Load Image ---
print(f"Loading image: {IMAGE_PATH}")
# OpenCV reads images in BGR format by default
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print(f"Error: Failed to load image '{IMAGE_PATH}'.")
    exit()
# Convert the image from BGR to RGB format (which SAM expects)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
print("Image loaded successfully.")

# --- Generate Masks ---
print("Generating masks... (This might take a while depending on image size and model)")
# The generate function returns a list of dictionaries, where each dictionary
# contains information about one detected object mask.
masks = mask_generator.generate(image_rgb)
print(f"Generated {len(masks)} masks.")

# --- Visualize and Save ---
if len(masks) > 0:
    print("Visualizing masks...")
    plt.figure(figsize=(12, 10))
    plt.imshow(image_rgb)  # Show the original image
    show_anns(masks, image_rgb)  # Overlay the colored masks
    plt.title(f"Automatic Segmentation - {len(masks)} masks found")
    plt.axis("off")  # Hide axes
    # Save the figure
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Segmented image saved to: {OUTPUT_IMAGE_PATH}")
    # Display the result in a window (optional)
    plt.show()
else:
    print("No masks were generated for this image.")

print("Script finished.")
