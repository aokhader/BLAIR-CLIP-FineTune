import json
import logging
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TFIDFBaseline:
    def __init__(self, data_dir='.', meta_file='meta_Appliances.jsonl', reviews_file='Appliances.jsonl', max_features=5000):
        self.data_dir = data_dir
        self.meta_file = os.path.join(data_dir, meta_file)
        self.reviews_file = os.path.join(data_dir, reviews_file)
        self.max_features = max_features
        
        self.item_texts = {}
        self.user_interactions = defaultdict(list)
        self.train_interactions = defaultdict(list)
        self.test_interactions = []
        self.user_vectors = {}
        self.tfidf_matrix = None
        self.asin_to_index = {}
        self.index_to_asin = {}
        self.vectorizer = None

    def load_jsonl(self, file_path):
        """Loads a JSONL file and yields objects."""
        with open(file_path, 'r') as f:
            for line in f:
                yield json.loads(line)

    def load_data(self):
        logging.info("Loading item metadata...")
        try:
            for obj in self.load_jsonl(self.meta_file):
                asin = obj.get('parent_asin')
                if not asin:
                    continue
                
                title = obj.get('title', '')
                description = " ".join(obj.get('description', []))
                features = " ".join(obj.get('features', []))
                
                full_text = f"{title} {description} {features}"
                self.item_texts[asin] = full_text
                
            logging.info(f"Loaded metadata for {len(self.item_texts)} items.")

        except FileNotFoundError:
            logging.error(f"File not found: {self.meta_file}")
            raise

        logging.info("Loading reviews...")
        try:
            for obj in self.load_jsonl(self.reviews_file):
                user_id = obj.get('user_id')
                asin = obj.get('parent_asin')
                timestamp = obj.get('timestamp')
                
                if user_id and asin and timestamp:
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
            
            if len(interactions) < 2:
                for _, asin in interactions:
                    self.train_interactions[user_id].append(asin)
                continue
                
            valid_users_count += 1
            
            test_item = interactions[-1][1]
            self.test_interactions.append((user_id, test_item))
            
            for _, asin in interactions[:-1]:
                self.train_interactions[user_id].append(asin)
                
        logging.info(f"Users with >= 2 interactions: {valid_users_count}")
        logging.info(f"Test set size: {len(self.test_interactions)}")

    def fit(self):
        logging.info("Vectorizing item texts...")
        all_asins = list(self.item_texts.keys())
        corpus = [self.item_texts[asin] for asin in all_asins]
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=self.max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        self.asin_to_index = {asin: i for i, asin in enumerate(all_asins)}
        self.index_to_asin = {i: asin for asin, i in self.asin_to_index.items()}
        
        logging.info(f"TF-IDF Matrix shape: {self.tfidf_matrix.shape}")

        logging.info("Computing user profiles...")
        count = 0
        for user_id, asins in self.train_interactions.items():
            valid_indices = [self.asin_to_index[a] for a in asins if a in self.asin_to_index]
            if not valid_indices:
                continue
                
            user_item_vectors = self.tfidf_matrix[valid_indices]
            user_vec = np.asarray(user_item_vectors.mean(axis=0))
            self.user_vectors[user_id] = user_vec
            
            count += 1
            if count % 100000 == 0:
                logging.info(f"Processed {count} users...")

        logging.info(f"Computed profiles for {len(self.user_vectors)} users.")

    def evaluate(self):
        logging.info("Evaluating (Ranking)...")
        
        recalls_at_10 = []
        recalls_at_50 = []
        aucs = []
        
        test_user_ids = [u for u, i in self.test_interactions if u in self.user_vectors and i in self.asin_to_index]
        test_ground_truths = [i for u, i in self.test_interactions if u in self.user_vectors and i in self.asin_to_index]
        
        logging.info(f"Evaluating on {len(test_user_ids)} valid test users...")
        
        processed = 0
        for i, user_id in enumerate(test_user_ids):
            gt_item = test_ground_truths[i]
            gt_index = self.asin_to_index[gt_item]
            
            user_vec = self.user_vectors[user_id]
            
            # Compute scores
            scores = self.tfidf_matrix.dot(user_vec.T).flatten()
            
            # Mask training items
            train_items = self.train_interactions[user_id]
            train_indices = [self.asin_to_index[a] for a in train_items if a in self.asin_to_index]
            
            scores[train_indices] = -np.inf
            
            # Ensure GT is not masked (if it was in train)
            if gt_index in train_indices:
                 scores[gt_index] = self.tfidf_matrix[gt_index].dot(user_vec.T)[0]

            gt_score = scores[gt_index]
            
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

    def run(self):
        logging.info("Starting TF-IDF + Cosine Similarity Baseline (Ranking Metrics)")
        self.load_data()
        self.preprocess()
        self.fit()
        return self.evaluate()

if __name__ == "__main__":
    # Assuming data is in the parent directory if running from tfidf_baseline folder,
    # or current directory if running from root.
    # The user ran from root previously.
    
    # Check if data files exist in current dir, else try parent
    if os.path.exists('meta_Appliances.jsonl'):
        data_dir = '.'
    elif os.path.exists('../meta_Appliances.jsonl'):
        data_dir = '..'
    else:
        # Default to current or absolute path if known
        data_dir = '.' 
        
    model = TFIDFBaseline(data_dir=data_dir)
    model.run()
