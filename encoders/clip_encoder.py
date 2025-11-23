"""
This module loads item images from a JSONL dataset, downloads them,
resizes them to 224x224, and converts them into CLIP image embeddings.

Used for:
    - CLIP baseline model (image-only)
    - BLaIR-MM fusion model (text + image)

Outputs:
    A single 512-dimensional CLIP embedding per item.
"""

import json
import requests
import torch
import clip
from PIL import Image
from io import BytesIO

# Load CLIP model + preprocess transform
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device="cpu")


def download_and_resize_image(url: str, size: tuple = (224, 224)) -> Image.Image | None:
    """
    Download an image from a URL and resize it.
    :param url: URL of the image.
    :param size: Desired output size for CLIP (224x224).
    :return: PIL Image or None if failed.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize(size, Image.BICUBIC)
        return img
    except Exception as e:
        print(f"Failed to process {url}: {e}")
        return None


def create_image_lists(dataset_path: str, max_items: int = 500000) -> dict:
    """
    Extract lists of PIL images (resized) for each item in the dataset.
    :param dataset_path: Path to JSONL file containing item metadata.
    :param max_items: Limit how many datapoints to process.
    :return: dict: asin -> list of PIL.Image objects
    """
    item_images = {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_items:
                break

            data = json.loads(line)
            asin = data.get("asin")
            if not asin:
                continue

            images = data.get("images", [])
            image_list = []

            for img_info in images:
                # Try in order of preference
                for key in [
                    "hi_res",
                    "large",
                    "thumb",
                    "large_image_url",
                    "medium_image_url",
                    "small_image_url",
                ]:
                    url = img_info.get(key)
                    if url:
                        img = download_and_resize_image(url)
                        if img:
                            image_list.append(img)

            if image_list:
                item_images[asin] = image_list

    return item_images


def encode_images_with_clip(image_list: list) -> torch.Tensor | None:
    """
    Takes a list of PIL images and returns a single fused CLIP embedding (512-dim).
    If multiple images are available, embeddings are averaged.
    :param image_list: List of PIL.Image objects.
    :return: torch.Tensor of shape [512] or None if no valid images.
    """
    if not image_list:
        return None

    embeddings = []

    for img in image_list:
        try:
            tensor_img = CLIP_PREPROCESS(img).unsqueeze(0)  # shape: [1, 3, 224, 224]

            with torch.no_grad():
                emb = CLIP_MODEL.encode_image(tensor_img)  # shape: [1, 512]
                emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
                embeddings.append(emb)

        except Exception as e:
            print(f"CLIP encoding failed: {e}")

    if not embeddings:
        return None

    # Average all embeddings for this item
    return torch.mean(torch.cat(embeddings, dim=0), dim=0)  # shape: [512]


if __name__ == "__main__":

    DATASET_PATH = "./training_datasets/meta_Appliances.jsonl"

    print("Extracting PIL images from dataset...")
    item_images = create_image_lists(DATASET_PATH, max_items=10)
    print(f"Extracted images for {len(item_images)} items.\n")

    print("Encoding images with CLIP...")
    clip_embeddings = {
        asin: encode_images_with_clip(img_list)
        for asin, img_list in item_images.items()
    }

    # Print results
    for asin, emb in clip_embeddings.items():
        print(f"\nASIN: {asin}")
        if emb is not None:
            print(f"CLIP embedding shape: {emb.shape}")  # should be torch.Size([512])
            print(f"Sample values: {emb[:5]}")
        else:
            print("No valid images found.")
