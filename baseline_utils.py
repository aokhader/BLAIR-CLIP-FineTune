import json
import logging
import os
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_jsonl(file_path):
    """Loads a JSONL file and yields objects."""
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)

class DataSplitter:
    def __init__(self, data_dir='.', meta_file='meta_Appliances.jsonl', reviews_file='Appliances.jsonl'):
        self.data_dir = data_dir
        self.meta_file = os.path.join(data_dir, meta_file)
        self.reviews_file = os.path.join(data_dir, reviews_file)
        
        self.item_texts = {}
        self.user_interactions = defaultdict(list)
        self.train_interactions = defaultdict(list)
        self.test_interactions = []
        self.all_asins = []
        self.asin_to_index = {}
        self.index_to_asin = {}

    def load_data(self):
        logging.info("Loading item metadata...")
        try:
            for obj in load_jsonl(self.meta_file):
                asin = obj.get('parent_asin')
                if not asin:
                    continue
                
                title = obj.get('title', '')
                description = " ".join(obj.get('description', []))
                features = " ".join(obj.get('features', []))
                
                full_text = f"{title} {description} {features}"
                self.item_texts[asin] = full_text
                
            self.all_asins = list(self.item_texts.keys())
            self.asin_to_index = {asin: i for i, asin in enumerate(self.all_asins)}
            self.index_to_asin = {i: asin for asin, i in self.asin_to_index.items()}
            
            logging.info(f"Loaded metadata for {len(self.item_texts)} items.")

        except FileNotFoundError:
            logging.error(f"File not found: {self.meta_file}")
            raise

        logging.info("Loading reviews...")
        try:
            for obj in load_jsonl(self.reviews_file):
                user_id = obj.get('user_id')
                asin = obj.get('parent_asin')
                timestamp = obj.get('timestamp')
                
                if user_id and asin and timestamp:
                    # Only keep interactions for items we have metadata for?
                    # Or keep all? TF-IDF baseline kept all but filtered later.
                    # Let's keep all for now, filter during split/train if needed.
                    self.user_interactions[user_id].append((timestamp, asin))
            
            logging.info(f"Loaded reviews for {len(self.user_interactions)} users.")
        except FileNotFoundError:
            logging.error(f"File not found: {self.reviews_file}")
            raise

    def preprocess(self):
        logging.info("Splitting data (Leave-One-Out)...")
        valid_users_count = 0
        
        for user_id, interactions in self.user_interactions.items():
            interactions.sort(key=lambda x: x[0])
            
            # Filter interactions to only include known items (if desired)
            # For TF-IDF, we need text, so we must filter.
            # For MF, we need consistent ID space.
            # Let's filter by known ASINs (from metadata) to ensure consistency.
            filtered_interactions = [(t, a) for t, a in interactions if a in self.asin_to_index]
            
            if len(filtered_interactions) < 2:
                # Put all in train if < 2
                for _, asin in filtered_interactions:
                    self.train_interactions[user_id].append(asin)
                continue
                
            valid_users_count += 1
            
            test_item = filtered_interactions[-1][1]
            self.test_interactions.append((user_id, test_item))
            
            for _, asin in filtered_interactions[:-1]:
                self.train_interactions[user_id].append(asin)
                
        logging.info(f"Users with >= 2 interactions (and valid items): {valid_users_count}")
        logging.info(f"Test set size: {len(self.test_interactions)}")

class Evaluator:
    def __init__(self, train_interactions, test_interactions, asin_to_index, n_items):
        self.train_interactions = train_interactions
        self.test_interactions = test_interactions
        self.asin_to_index = asin_to_index
        self.n_items = n_items

    def evaluate(self, score_func, user_vectors=None, item_vectors=None, batch_size=1000):
        """
        score_func: function(user_id) -> numpy array of scores (N_items,)
                    OR function(user_ids) -> numpy array (Batch, N_items)
        """
        logging.info("Evaluating (Ranking)...")
        
        recalls_at_10 = []
        recalls_at_50 = []
        aucs = []
        
        # Filter test users who are in train (cold start users in test might fail if no embedding)
        # For TF-IDF, we computed user vector from train items.
        # For MF, we have user embeddings.
        
        # We assume score_func handles user_id validity or we filter here.
        valid_test_users = [u for u, i in self.test_interactions if u in self.train_interactions]
        # Actually, for TF-IDF, a user might have 0 train items if we filtered them out? 
        # But we filtered in preprocess.
        
        # For MF, we need user index. We'll handle mapping in the model wrapper.
        
        test_data = [(u, i) for u, i in self.test_interactions if u in self.train_interactions]
        logging.info(f"Evaluating on {len(test_data)} valid test users...")
        
        processed = 0
        for i, (user_id, gt_item) in enumerate(test_data):
            gt_index = self.asin_to_index[gt_item]
            
            # Get scores for all items
            scores = score_func(user_id) # Should return (N_items,)
            
            # Mask training items
            train_items = self.train_interactions[user_id]
            train_indices = [self.asin_to_index[a] for a in train_items if a in self.asin_to_index]
            
            scores[train_indices] = -np.inf
            
            # Ensure GT is not masked
            # (If GT was in train, it means re-interaction. We unmask it to rank it.)
            # Note: score_func might need to re-compute score for GT if we just overwrote it with -inf
            # But usually we just set -inf. If GT index is in train_indices, we need to restore it.
            # Ideally, score_func returns pure scores. We modify a copy or modify in place carefully.
            
            if gt_index in train_indices:
                # We need the original score. 
                # Optimization: Don't overwrite GT index.
                # Remove GT index from train_indices if present
                if gt_index in train_indices:
                    # This check is redundant but safe
                    pass
            
            # Better way:
            # scores[train_indices] = -np.inf
            # scores[gt_index] = original_score 
            # But we lost original score.
            
            gt_score = scores[gt_index]
            
            # Now mask
            scores[train_indices] = -np.inf
            scores[gt_index] = gt_score # Restore GT score
            
            # Rank
            higher_scores = (scores > gt_score).sum()
            rank = higher_scores + 1
            
            recalls_at_10.append(1 if rank <= 10 else 0)
            recalls_at_50.append(1 if rank <= 50 else 0)
            
            valid_count = (scores > -np.inf).sum()
            num_negatives = valid_count - 1
            
            if num_negatives > 0:
                auc = 1.0 - (rank - 1) / num_negatives
                aucs.append(auc)
            
            processed += 1
            if processed % 1000 == 0:
                logging.info(f"Evaluated {processed} users. Current R@10: {np.mean(recalls_at_10):.4f}")

        avg_r10 = np.mean(recalls_at_10)
        avg_r50 = np.mean(recalls_at_50)
        avg_auc = np.mean(aucs)
        
        logging.info("Final Results:")
        logging.info(f"Recall@10: {avg_r10:.4f}")
        logging.info(f"Recall@50: {avg_r50:.4f}")
        logging.info(f"AUC:       {avg_auc:.4f}")
        
        return {
            'Recall@10': avg_r10,
            'Recall@50': avg_r50,
            'AUC': avg_auc
        }
