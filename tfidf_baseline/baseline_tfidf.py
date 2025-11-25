import json
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_jsonl(file_path):
    """Loads a JSONL file and yields objects."""
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)

def main():
    logging.info("Starting TF-IDF + Cosine Similarity Baseline (Ranking Metrics)")

    # 1. Load Metadata (Items)
    logging.info("Loading item metadata...")
    item_texts = {} # asin -> text
    
    meta_file = 'meta_Appliances.jsonl'
    reviews_file = 'Appliances.jsonl'

    try:
        for obj in load_jsonl(meta_file):
            # Use parent_asin as the identifier since asin is missing in metadata
            asin = obj.get('parent_asin')
            if not asin:
                continue
            
            title = obj.get('title', '')
            description = " ".join(obj.get('description', []))
            features = " ".join(obj.get('features', []))
            
            # Combine text fields
            full_text = f"{title} {description} {features}"
            item_texts[asin] = full_text
            
        logging.info(f"Loaded metadata for {len(item_texts)} items.")

    except FileNotFoundError:
        logging.error(f"File not found: {meta_file}")
        return

    # 2. Load Reviews (Interactions) with Timestamps
    logging.info("Loading reviews...")
    user_interactions = defaultdict(list) # user_id -> list of (timestamp, asin)
    
    try:
        for obj in load_jsonl(reviews_file):
            user_id = obj.get('user_id')
            # Use parent_asin to match metadata
            asin = obj.get('parent_asin')
            timestamp = obj.get('timestamp')
            
            if user_id and asin and timestamp:
                user_interactions[user_id].append((timestamp, asin))
        
        logging.info(f"Loaded reviews for {len(user_interactions)} users.")
    except FileNotFoundError:
        logging.error(f"File not found: {reviews_file}")
        return

    # 3. Split Data (Leave-One-Out)
    logging.info("Splitting data (Leave-One-Out)...")
    train_interactions = defaultdict(list)
    test_interactions = [] # List of (user_id, ground_truth_asin)
    
    # Filter users with at least 2 interactions (1 for train, 1 for test)
    valid_users_count = 0
    
    for user_id, interactions in user_interactions.items():
        # Sort by timestamp
        interactions.sort(key=lambda x: x[0])
        
        if len(interactions) < 2:
            for _, asin in interactions:
                train_interactions[user_id].append(asin)
            continue
            
        valid_users_count += 1
        
        # Last one is test
        test_item = interactions[-1][1]
        test_interactions.append((user_id, test_item))
        
        # Rest are train
        for _, asin in interactions[:-1]:
            train_interactions[user_id].append(asin)
            
    logging.info(f"Users with >= 2 interactions: {valid_users_count}")
    logging.info(f"Test set size: {len(test_interactions)}")

    # 4. TF-IDF Vectorization of Items
    logging.info("Vectorizing item texts...")
    all_asins = list(item_texts.keys())
    corpus = [item_texts[asin] for asin in all_asins]
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Map ASIN to index in tfidf_matrix
    asin_to_index = {asin: i for i, asin in enumerate(all_asins)}
    index_to_asin = {i: asin for asin, i in asin_to_index.items()}
    
    logging.info(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

    # 5. Compute User Vectors (Train Set)
    logging.info("Computing user profiles...")
    user_vectors = {} # user_id -> numpy array
    
    count = 0
    for user_id, asins in train_interactions.items():
        valid_indices = [asin_to_index[a] for a in asins if a in asin_to_index]
        if not valid_indices:
            continue
            
        # Extract rows for these items
        user_item_vectors = tfidf_matrix[valid_indices]
        
        # Compute mean vector
        user_vec = np.asarray(user_item_vectors.mean(axis=0))
        user_vectors[user_id] = user_vec
        
        count += 1
        if count % 100000 == 0:
            logging.info(f"Processed {count} users...")

    logging.info(f"Computed profiles for {len(user_vectors)} users.")

    # 6. Prediction & Evaluation (Ranking)
    logging.info("Evaluating (Ranking)...")
    
    recalls_at_10 = []
    recalls_at_50 = []
    aucs = []
    
    # Pre-compute all item vectors for fast scoring
    # tfidf_matrix is (N_items, F). User vec is (1, F).
    # Score = UserVec @ ItemMatrix.T -> (1, N_items)
    # This is efficient.
    
    # Optimization: Process in batches of users
    test_user_ids = [u for u, i in test_interactions if u in user_vectors and i in asin_to_index]
    test_ground_truths = [i for u, i in test_interactions if u in user_vectors and i in asin_to_index]
    
    logging.info(f"Evaluating on {len(test_user_ids)} valid test users (with history and valid items)...")
    
    processed = 0
    for i, user_id in enumerate(test_user_ids):
        gt_item = test_ground_truths[i]
        gt_index = asin_to_index[gt_item]
        
        user_vec = user_vectors[user_id] # (1, F)
        
        # Compute scores for ALL items
        scores = tfidf_matrix.dot(user_vec.T).flatten() # (N,)
        
        # Mask training items
        train_items = train_interactions[user_id]
        train_indices = [asin_to_index[a] for a in train_items if a in asin_to_index]
        
        # Set train items scores to -inf
        scores[train_indices] = -np.inf
        
        # Ensure GT score is preserved
        if gt_index in train_indices:
             scores[gt_index] = tfidf_matrix[gt_index].dot(user_vec.T)[0]

        # Get rank of GT item
        gt_score = scores[gt_index]
        
        higher_scores = (scores > gt_score).sum()
        rank = higher_scores + 1
        
        # Recall@K
        recalls_at_10.append(1 if rank <= 10 else 0)
        recalls_at_50.append(1 if rank <= 50 else 0)
        
        # AUC = 1 - (Rank - 1) / TotalNegatives
        valid_count = (scores > -np.inf).sum()
        num_negatives = valid_count - 1
        
        if num_negatives > 0:
            auc = 1.0 - (rank - 1) / num_negatives
            aucs.append(auc)
        
        processed += 1
        if processed % 1000 == 0:
            logging.info(f"Evaluated {processed} users. Current R@10: {np.mean(recalls_at_10):.4f}")

    # 7. Final Metrics
    avg_r10 = np.mean(recalls_at_10)
    avg_r50 = np.mean(recalls_at_50)
    avg_auc = np.mean(aucs)
    
    logging.info("Final Results:")
    logging.info(f"Recall@10: {avg_r10:.4f}")
    logging.info(f"Recall@50: {avg_r50:.4f}")
    logging.info(f"AUC:       {avg_auc:.4f}")
    logging.info("Done.")

if __name__ == "__main__":
    main()
