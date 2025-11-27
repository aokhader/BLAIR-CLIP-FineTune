"""
TF-IDF Baseline Model
---------------------

This baseline performs next-item recommendation using:
    - TF-IDF text embeddings from item metadata
    - Mean user profile vectors (average of TF-IDF vectors of items the user has interacted with)
    - Cosine similarity scoring (via dot products)

The model uses the BLaIR benchmark split files:
    release_amazon/0core/last_out/*.csv

Files required:
    - meta_{CATEGORY}.jsonl  (contains item titles, descriptions, features)
    - {CATEGORY}.train.csv
    - {CATEGORY}.valid.csv   (not used, but loaded)
    - {CATEGORY}.test.csv

Outputs:
    Recall@10, Recall@50, AUC
"""

import os
import json
import csv
import logging
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TFIDFBaseline:
    def __init__(
        self,
        data_dir: str,
        prefix: str,
        meta_file: str,
        max_features: int = 5000,
    ):
        """
        :param data_dir: Root directory containing release_amazon/0core/
        :param prefix: Category prefix, e.g., "Appliances"
        :param meta_file: Path to metadata JSONL file
        """
        self.data_dir = data_dir
        self.prefix = prefix
        self.meta_path = os.path.join(data_dir, meta_file)

        # Benchmark split paths
        base = os.path.join(data_dir, "release_amazon", "0core", "last_out")
        self.train_path = os.path.join(base, f"{prefix}.train.csv")
        self.valid_path = os.path.join(base, f"{prefix}.valid.csv")
        self.test_path = os.path.join(base, f"{prefix}.test.csv")

        # Storage
        self.item_texts: Dict[str, str] = {}
        self.train_interactions: Dict[str, List[str]] = defaultdict(list)
        self.test_interactions: List[Tuple[str, str]] = []

        # Vectorization objects
        self.vectorizer = None
        self.tfidf_matrix = None
        self.asin_to_index = {}
        self.index_to_asin = {}
        self.user_vectors = {}

        self.max_features = max_features

    # ---------------------------------------------------------------
    # Loading Helpers
    # ---------------------------------------------------------------
    def load_jsonl(self, file_path):
        """Generator to load JSONL files."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    def load_metadata_texts(self):
        """Load item metadata and construct text fields."""
        logging.info("Loading metadata text fields...")

        for obj in self.load_jsonl(self.meta_path):
            asin = obj.get("parent_asin")
            if not asin:
                continue

            title = obj.get("title", "")
            description = " ".join(obj.get("description", []))
            features = " ".join(obj.get("features", []))

            full_text = f"{title} {description} {features}".strip()
            self.item_texts[asin] = full_text

        logging.info(f"Loaded text for {len(self.item_texts)} items.")

    def load_splits(self):
        """Load BLaIR train/test split CSV files."""
        logging.info("Loading split CSVs...")

        # --- TRAIN ---
        with open(self.train_path, "r") as f:
            reader = csv.reader(f)
            for user, asin, _, _ in reader:
                self.train_interactions[user].append(asin)

        # --- TEST ---
        with open(self.test_path, "r") as f:
            reader = csv.reader(f)
            for user, asin, _, _ in reader:
                self.test_interactions.append((user, asin))

        logging.info(f"Train users: {len(self.train_interactions)}")
        logging.info(f"Test samples: {len(self.test_interactions)}")

    # ---------------------------------------------------------------
    # TF-IDF Vectorization
    # ---------------------------------------------------------------
    def build_tfidf_matrix(self):
        """Build TF-IDF vectors for all item texts."""
        logging.info("Vectorizing item texts with TF-IDF...")

        all_asins = list(self.item_texts.keys())
        corpus = [self.item_texts[a] for a in all_asins]

        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=self.max_features
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        self.asin_to_index = {asin: i for i, asin in enumerate(all_asins)}
        self.index_to_asin = {i: asin for asin, i in self.asin_to_index.items()}

        logging.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    # ---------------------------------------------------------------
    # User Profile Construction
    # ---------------------------------------------------------------
    def build_user_profiles(self):
        """Average TF-IDF vectors of training items for each user."""
        logging.info("Building user vectors...")

        for user, items in self.train_interactions.items():
            valid_idxs = [
                self.asin_to_index[i] for i in items if i in self.asin_to_index
            ]

            if not valid_idxs:
                continue

            vectors = self.tfidf_matrix[valid_idxs].toarray()
            self.user_vectors[user] = vectors.mean(axis=0)

        logging.info(f"Built {len(self.user_vectors)} user profiles.")

    # ---------------------------------------------------------------
    # Evaluation (Recall@K + AUC)
    # ---------------------------------------------------------------
    def evaluate(self):
        logging.info("Evaluating TF-IDF baseline...")
        recalls_at_10, recalls_at_50, aucs = [], [], []

        valid_tests = [
            (u, i)
            for (u, i) in self.test_interactions
            if u in self.user_vectors and i in self.asin_to_index
        ]

        logging.info(f"Valid test samples: {len(valid_tests)}")

        for user, gt_asin in valid_tests:
            user_vec = self.user_vectors[user]

            # Cosine similarity = dot product (vectors are L2-normalized implicitly)
            scores = self.tfidf_matrix.dot(user_vec).flatten()

            gt_idx = self.asin_to_index[gt_asin]
            gt_score = scores[gt_idx]

            # Compute rank
            rank = (scores > gt_score).sum() + 1

            recalls_at_10.append(1 if rank <= 10 else 0)
            recalls_at_50.append(1 if rank <= 50 else 0)

            num_items = len(scores)
            auc = 1.0 - ((rank - 1) / (num_items - 1))
            aucs.append(auc)

        # Compute final metrics
        r10 = np.mean(recalls_at_10)
        r50 = np.mean(recalls_at_50)
        auc = np.mean(aucs)

        logging.info(
            f"TF-IDF Results — Recall@10 = {r10:.4f}, Recall@50 = {r50:.4f}, AUC = {auc:.4f}"
        )

        return {"Recall@10": r10, "Recall@50": r50, "AUC": auc}

    # ---------------------------------------------------------------
    # RUN PIPELINE
    # ---------------------------------------------------------------
    def run(self):
        logging.info("Running TF-IDF + Cosine Similarity Baseline...")

        self.load_metadata_texts()
        self.load_splits()
        self.build_tfidf_matrix()
        self.build_user_profiles()

        return self.evaluate()


# ---------------------------------------------------------------
# Main (Manual Debug Run)
# ---------------------------------------------------------------
if __name__ == "__main__":
    model = TFIDFBaseline(
        data_dir=".",  # root folder
        prefix="Appliances_No_Images",  # category name
        meta_file="meta/meta_Appliances.json",
    )

    results = model.run()
    print("\nFinal TF-IDF Baseline Metrics:", results)
