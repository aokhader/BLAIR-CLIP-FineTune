import argparse
import hashlib
import json
import logging
import os
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from datasets import load_dataset
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

DEFAULT_CATEGORIES = ["Appliances"]
NUM_WORKERS = 1
VALID_TIMESTAMP = 1628643414042
DOWNSAMPLING_FACTOR = 10
MIN_TEXT_LENGTH = 30

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (compatible; Blair-MM/0.1; +https://github.com/hyp1231/BLAIR)",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    }
)
retry_strategy = Retry(
    total=5,
    read=5,
    connect=3,
    backoff_factor=1.2,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
)
adapter = HTTPAdapter(max_retries=retry_strategy)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare multimodal BLaIR pretraining data.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Amazon categories to include (defaults to Appliances only).",
    )
    parser.add_argument(
        "--output_tsv",
        default="clean_review_meta_with_images.tsv",
        help="Destination TSV containing review/meta/image columns.",
    )
    parser.add_argument(
        "--image_dir",
        default="blair_clip_images",
        help="Directory where resized product images will be stored.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Square resolution used when resizing images for CLIP.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Optional limit on number of metadata entries (useful for debugging).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for the downsampling process.",
    )
    parser.add_argument(
        "--metadata_cache",
        default="metadata_cache.json",
        help="Path to JSON cache storing downloaded metadata + image paths for faster resume.",
    )
    parser.add_argument(
        "--resume_output",
        action="store_true",
        help="If set and output TSV exists, keep existing rows and skip duplicates.",
    )
    return parser.parse_args()


def concat_item_metadata(dp):
    meta_segments: List[str] = []
    if dp.get("title"):
        meta_segments.append(dp["title"])
    if dp.get("features"):
        meta_segments.extend(dp["features"])
    if dp.get("description"):
        meta_segments.extend(dp["description"])
    dp["cleaned_metadata"] = (
        " ".join(meta_segments).replace("\t", " ").replace("\n", " ").replace("\r", "").strip()
    )
    return dp


def concat_review(dp):
    review_segments: List[str] = []
    if dp.get("title"):
        review_segments.append(dp["title"])
    if dp.get("text"):
        review_segments.append(dp["text"])
    dp["cleaned_review"] = (
        " ".join(review_segments).replace("\t", " ").replace("\n", " ").replace("\r", "").strip()
    )
    return dp


def select_image_url(images: Dict[str, List[Optional[str]]]) -> Optional[str]:
    # Prioritize higher quality variants when available.
    priority_keys = ["hi_res", "large", "thumb"]
    for key in priority_keys:
        candidates = images.get(key) if isinstance(images, dict) else None
        if not candidates:
            continue
        for url in candidates:
            if url:
                return url
    return None


def download_and_resize_image(
    url: str,
    asin: str,
    image_dir: str,
    image_size: int,
) -> Optional[str]:
    os.makedirs(image_dir, exist_ok=True)
    filename = f"{asin}.jpg"
    path = os.path.join(image_dir, filename)
    try:
        response = SESSION.get(url, timeout=(5, 30))
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize((image_size, image_size), Image.BICUBIC)
        image.save(path, format="JPEG")
        return path
    except Exception as exc:  # noqa: BLE001 - best effort download
        logger.warning("Failed to fetch image %s: %s", url, exc)
        return None


def load_metadata_cache(cache_path: Optional[str]) -> Dict[str, Tuple[str, str]]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cache: Dict[str, Tuple[str, str]] = {}
        for asin, values in payload.items():
            meta = values.get("meta")
            image_path = values.get("image_path")
            if meta and image_path and os.path.exists(image_path):
                cache[asin] = (meta, image_path)
        logger.info("Loaded %d cached metadata entries from %s", len(cache), cache_path)
        return cache
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load metadata cache %s: %s", cache_path, exc)
        return {}


def save_metadata_cache(cache_path: Optional[str], store: Dict[str, Tuple[str, str]]):
    if not cache_path:
        return
    serializable = {
        asin: {"meta": meta_text, "image_path": image_path}
        for asin, (meta_text, image_path) in store.items()
    }
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle)
    os.replace(tmp_path, cache_path)


def pair_hash(review: str, meta: str) -> str:
    return hashlib.md5((review + "\n" + meta).encode("utf-8")).hexdigest()


def build_metadata_store(
    categories: List[str],
    image_dir: str,
    image_size: int,
    max_items: Optional[int],
    initial_store: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, Tuple[str, str]]:
    store: Dict[str, Tuple[str, str]] = dict(initial_store or {})
    processed = len(store)
    if processed:
        logger.info("Starting with %d cached metadata entries", processed)
    for category in categories:
        logger.info("Loading metadata for category %s", category)
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            # trust_remote_code=True,
        )
        logger.info("Loaded %d metadata entries for category %s", len(meta_dataset), category)
        concat_dataset = meta_dataset.map(concat_item_metadata, num_proc=NUM_WORKERS)
        filtered_dataset = concat_dataset.filter(
            lambda dp: len(dp["cleaned_metadata"]) > MIN_TEXT_LENGTH,
            num_proc=NUM_WORKERS,
        )
        for idx, dp in enumerate(filtered_dataset, start=1):
            asin = dp["parent_asin"]
            if asin in store:
                continue
            image_url = select_image_url(dp.get("images", {}))
            if not image_url:
                continue
            print(image_url)
            image_path = download_and_resize_image(
                image_url,
                asin,
                image_dir=image_dir,
                image_size=image_size,
            )
            if not image_path:
                continue
            store[asin] = (dp["cleaned_metadata"], image_path)
            processed += 1
            if processed % 500 == 0:
                logger.info("Stored %d items with images so far (category=%s, seen=%d)", processed, category, idx)
            if max_items and processed >= max_items:
                logger.info("Reached max metadata limit of %d entries", max_items)
                return store
    logger.info("Collected %d metadata entries with images.", len(store))
    return store


def filter_reviews(dp, metadata_store: Dict[str, Tuple[str, str]]) -> bool:
    if random.randint(1, DOWNSAMPLING_FACTOR) > 1:
        return False
    if dp["timestamp"] >= VALID_TIMESTAMP:
        return False
    asin = dp["parent_asin"]
    if asin not in metadata_store:
        return False
    if len(dp["cleaned_review"]) <= MIN_TEXT_LENGTH:
        return False
    return True


def main():
    args = parse_args()
    random.seed(args.seed)

    cached_store = load_metadata_cache(args.metadata_cache)
    metadata_store = build_metadata_store(
        categories=args.categories,
        image_dir=args.image_dir,
        image_size=args.image_size,
        max_items=args.max_items,
        initial_store=cached_store,
    )
    save_metadata_cache(args.metadata_cache, metadata_store)

    reviews: List[str] = []
    metas: List[str] = []
    image_paths: List[str] = []
    seen_hashes = set()
    if args.resume_output and os.path.exists(args.output_tsv):
        try:
            existing_df = pd.read_csv(args.output_tsv, sep="\t")
            reviews = existing_df["review"].tolist()
            metas = existing_df["meta"].tolist()
            image_paths = existing_df["image_path"].tolist()
            for review_text, meta_text in zip(reviews, metas):
                seen_hashes.add(pair_hash(review_text, meta_text))
            logger.info("Resumed with %d existing triples from %s", len(reviews), args.output_tsv)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to resume from %s: %s", args.output_tsv, exc)

    for category in args.categories:
        logger.info("Loading reviews for category %s", category)
        review_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split="full",
            # trust_remote_code=True,
        )
        concat_reviews = review_dataset.map(concat_review, num_proc=NUM_WORKERS)
        filtered_reviews = concat_reviews.filter(
            lambda dp: filter_reviews(dp, metadata_store),
            num_proc=NUM_WORKERS,
        )
        for idx, (review_text, asin) in enumerate(
            zip(
                filtered_reviews["cleaned_review"],
                filtered_reviews["parent_asin"],
            ),
            start=1,
        ):
            meta_text, local_image_path = metadata_store[asin]
            review_meta_hash = pair_hash(review_text, meta_text)
            if review_meta_hash in seen_hashes:
                continue
            reviews.append(review_text)
            metas.append(meta_text)
            image_paths.append(local_image_path)
            seen_hashes.add(review_meta_hash)
            if len(reviews) % 500 == 0:
                logger.info(
                    "Prepared %d review/meta/image triples so far (category=%s, seen=%d)",
                    len(reviews),
                    category,
                    idx,
                )

    df = pd.DataFrame(
        {
            "review": reviews,
            "meta": metas,
            "image_path": image_paths,
        }
    )
    df.to_csv(args.output_tsv, sep="\t", lineterminator="\n", index=False)
    logger.info(
        "Saved %d multimodal training pairs with images to %s",
        len(df),
        args.output_tsv,
    )


if __name__ == "__main__":
    main()

