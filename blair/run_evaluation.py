#!/usr/bin/env python3
import os
import csv
import logging
import json
import sys
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from transformers import (
    AutoTokenizer, 
    CLIPImageProcessor, 
    RobertaModel,
    BertModel,
    AutoConfig,
    AutoModel
)

# Add current directory to path to find multimodal module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multimodal.blair_clip import BlairCLIPDualEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomDataSplitter:
    def __init__(self, data_dir='splits/with_images', 
                 meta_file='./clean_review_meta_with_images.tsv'):
        self.data_dir = data_dir
        self.meta_file = meta_file
        
        self.train_file = os.path.join(data_dir, 'Appliances.train.csv')
        self.test_file = os.path.join(data_dir, 'Appliances.test.csv')
        self.valid_file = os.path.join(data_dir, 'Appliances.valid.csv')
        
        self.item_texts = {}
        self.item_images = {} # ASIN -> image_path
        self.train_interactions = defaultdict(list)
        self.test_interactions = []
        
        self.all_asins = set()
        self.asin_to_index = {}
        self.index_to_asin = {}

    def load_data(self):
        logging.info("Loading metadata from TSV...")
        try:
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                header = next(reader) # review, meta, image_path
                
                try:
                    meta_idx = header.index('meta')
                    img_idx = header.index('image_path')
                except ValueError:
                    meta_idx = 1
                    img_idx = 2
                
                for row in reader:
                    if len(row) <= max(meta_idx, img_idx):
                        continue
                        
                    meta_text = row[meta_idx]
                    image_path = row[img_idx]
                    
                    basename = os.path.basename(image_path)
                    asin = os.path.splitext(basename)[0]
                    
                    if asin:
                        self.item_texts[asin] = meta_text
                        self.item_images[asin] = image_path
                        self.all_asins.add(asin)
                        
            logging.info(f"Loaded metadata for {len(self.item_texts)} items.")
            
        except FileNotFoundError:
            logging.error(f"File not found: {self.meta_file}")
            raise

        logging.info("Loading interactions from CSVs...")
        self._load_interactions(self.train_file, is_train=True)
        self._load_interactions(self.test_file, is_train=False)
        self._load_interactions(self.valid_file, is_train=True) # Add valid to history
        
        self.all_asins = sorted(list(self.all_asins))
        self.asin_to_index = {asin: i for i, asin in enumerate(self.all_asins)}
        self.index_to_asin = {i: asin for asin, i in self.asin_to_index.items()}
        
        logging.info(f"Total Items: {len(self.all_asins)}")
        logging.info(f"Train Users: {len(self.train_interactions)}")
        logging.info(f"Test Pairs: {len(self.test_interactions)}")

    def _load_interactions(self, file_path, is_train=True):
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            return
            
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                user_id = row[0]
                asin = row[1]
                
                if asin not in self.all_asins:
                    self.all_asins.add(asin)
                    self.item_texts[asin] = ""
                    self.item_images[asin] = ""
                
                if is_train:
                    self.train_interactions[user_id].append(asin)
                else:
                    self.test_interactions.append((user_id, asin))

class ItemDataset(Dataset):
    def __init__(self, items, item_texts, item_images, tokenizer, image_processor, max_len=64, image_root="."):
        self.items = items
        self.item_texts = item_texts
        self.item_images = item_images
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        self.image_root = image_root

    def __len__(self):
        return len(self.items)
        
    def __getitem__(self, idx):
        asin = self.items[idx]
        text = self.item_texts.get(asin, "")
        img_path = self.item_images.get(asin, "")
        
        text_inputs = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        pixel_values = torch.zeros(1) # Default dummy
        
        if self.image_processor is not None:
            pixel_values = None
            if img_path:
                try:
                    full_path = os.path.join(self.image_root, img_path)
                    image = Image.open(full_path).convert("RGB")
                    pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
                except Exception:
                    pass
            
            if pixel_values is None:
                # Black image
                if hasattr(self.image_processor, 'crop_size'):
                    size = self.image_processor.crop_size
                    h, w = size['height'], size['width']
                else:
                    h, w = 224, 224
                    
                image = Image.new("RGB", (w, h), color=0)
                pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': pixel_values
        }

class BlairEvaluator:
    def __init__(self, model_path, data_splitter, batch_size=64, device='cuda', model_type='blair_clip'):
        self.model_path = model_path
        self.splitter = data_splitter
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.text_embeddings = None
        self.image_embeddings = None

    def load_model(self):
        logging.info(f"Loading model ({self.model_type}) from {self.model_path}...")
        
        if self.model_type == 'blair_clip':
            # Initialize Base Objects
            # Assuming defaults from base_clip.sh: roberta-base, clip-vit-base-patch16, 512 dim
            text_model_name = "roberta-base"
            clip_model_name = "openai/clip-vit-base-patch16"
            projection_dim = 512
            
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
            
            # Build Model Structure
            text_encoder = RobertaModel.from_pretrained(text_model_name)
            self.model = BlairCLIPDualEncoder(
                text_encoder=text_encoder,
                pooler_type="cls",
                projection_dim=projection_dim,
                clip_model_name=clip_model_name,
                # Init params that affect structure
            )
            
            # Load State Dict
            weights_path = os.path.join(self.model_path, "pytorch_model.bin")
            if not os.path.exists(weights_path):
                 # Try without .bin or looking for checkpoint
                 raise FileNotFoundError(f"Model weights not found at {weights_path}")
                 
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Handle potential key mismatches if any (e.g. DDP prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load weights
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            logging.info(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
        elif self.model_type in ['blair_base', 'blair_large']:
            # Load HF model directly
            # If model_path is a path, use it. If it's the HF ID, use it.
            load_path = self.model_path
            
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = AutoModel.from_pretrained(load_path)
            self.image_processor = None # No images
            
        self.model.to(self.device)
        self.model.eval()
        if self.device != 'cpu':
             self.model.half() # Convert to half precision

    def encode_all_items(self, cache_file="item_embeddings_dict.pt", force_refresh=False):
        # Unique cache file per model type
        # Use safe name
        safe_model_name = os.path.basename(self.model_path) if os.path.exists(self.model_path) else self.model_path.replace("/", "_")
        base, ext = os.path.splitext(cache_file)
        cache_file = f"{base}_{safe_model_name}{ext}"

        if not force_refresh and os.path.exists(cache_file):
            logging.info(f"Loading cached item embeddings from {cache_file}...")
            data = torch.load(cache_file, map_location='cpu')
            if isinstance(data, dict):
                self.text_embeddings = data['text']
                self.image_embeddings = data['image']
                logging.info(f"Loaded text/image embeddings for {len(self.text_embeddings)} items.")
                return
            else:
                logging.warning("Cache file format mismatch (expected dict), re-encoding...")

        logging.info(f"Encoding all items (Model: {self.model_type})...")
        
        dataset = ItemDataset(
            self.splitter.all_asins,
            self.splitter.item_texts,
            self.splitter.item_images,
            self.tokenizer,
            self.image_processor,
            image_root="." # Assuming images are in current dir or handled
        )
        
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)
        
        all_text = []
        all_image = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding Items"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                
                if self.device != 'cpu':
                    if pixel_values.numel() > 1: # If not dummy scalar
                        pixel_values = pixel_values.half()
                
                if self.model_type == 'blair_clip':
                    text_embeds = self.model.encode_text(input_ids, attention_mask=attention_mask)
                    image_embeds = self.model.encode_image(pixel_values)
                else:
                    # HF AutoModel (RoBERTa)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    # Use [CLS] embedding (first token)
                    text_embeds = outputs.last_hidden_state[:, 0]
                    text_embeds = F.normalize(text_embeds, dim=-1)
                    # Dummy image embeddings
                    image_embeds = torch.zeros_like(text_embeds)

                all_text.append(text_embeds.cpu())
                all_image.append(image_embeds.cpu())
                
        self.text_embeddings = torch.cat(all_text, dim=0) # (N_items, Dim)
        self.image_embeddings = torch.cat(all_image, dim=0) # (N_items, Dim)
        
        logging.info(f"Encoded {len(self.text_embeddings)} items.")
        
        logging.info(f"Saving item embeddings to {cache_file}...")
        torch.save({
            'text': self.text_embeddings,
            'image': self.image_embeddings
        }, cache_file)

    def evaluate(self, limit=None, mode='combined'):
        if self.model_type != 'blair_clip' and mode in ['image', 'combined']:
            logging.info(f"Skipping evaluation mode '{mode}' for text-only model '{self.model_type}'.")
            return None

        logging.info(f"Evaluating Mode: {mode} (Ranking)...")
        
        recalls_at_10 = []
        recalls_at_50 = []
        mrrs = []
        ndcg_at_10 = []
        ndcg_at_50 = []
        aucs = []
        
        # Select Embedding Matrix
        if mode == 'text':
            item_embeddings = self.text_embeddings
        elif mode == 'image':
            item_embeddings = self.image_embeddings
        elif mode == 'combined':
            item_embeddings = (self.text_embeddings + self.image_embeddings) / 2.0
            item_embeddings = F.normalize(item_embeddings, dim=-1)
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")
        
        # Filter test users who have training history
        valid_test_pairs = [
            (u, i) for u, i in self.splitter.test_interactions 
            if u in self.splitter.train_interactions and i in self.splitter.asin_to_index
        ]
        
        if limit:
            logging.info(f"Limiting evaluation to first {limit} users")
            valid_test_pairs = valid_test_pairs[:limit]
        
        logging.info(f"Evaluating on {len(valid_test_pairs)} valid test pairs...")
        
        # We can batch users for speed, but let's do loop for simplicity first
        # Pre-compute user embeddings
        
        # Map user -> train item indices
        user_history_indices = {}
        for u in self.splitter.train_interactions:
            indices = [self.splitter.asin_to_index[i] for i in self.splitter.train_interactions[u] if i in self.splitter.asin_to_index]
            if indices:
                user_history_indices[u] = indices
        
        item_embeddings = item_embeddings.to(self.device)
        
        processed = 0
        for user_id, gt_asin in tqdm(valid_test_pairs, desc=f"Evaluating Users ({mode})"):
            if user_id not in user_history_indices:
                continue
                
            hist_indices = user_history_indices[user_id]
            gt_index = self.splitter.asin_to_index[gt_asin]
            
            # Compute User Vector (Mean of history items)
            # (NumHistory, Dim)
            hist_vecs = item_embeddings[hist_indices]
            user_vec = torch.mean(hist_vecs, dim=0, keepdim=True) # (1, Dim)
            user_vec = F.normalize(user_vec, dim=-1)
            
            # Compute Scores: User @ AllItems.T
            # (1, Dim) @ (Dim, N_items) -> (1, N_items)
            scores = torch.mm(user_vec, item_embeddings.t()).squeeze(0) # (N_items)
            
            # Mask training items
            scores[hist_indices] = -float('inf')
            
            if gt_index in hist_indices:
                # Re-calculate GT score specifically
                gt_score_val = torch.mm(user_vec, item_embeddings[gt_index].unsqueeze(1)).item()
                scores[gt_index] = gt_score_val
            
            gt_score = scores[gt_index].item()
            
            # Rank
            higher_scores = (scores > gt_score).sum().item()
            rank = higher_scores + 1
            
            # Recall
            recalls_at_10.append(1 if rank <= 10 else 0)
            recalls_at_50.append(1 if rank <= 50 else 0)
            
            # MRR
            mrrs.append(1.0 / rank)
            
            # NDCG
            ndcg_at_10.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)
            ndcg_at_50.append(1.0 / np.log2(rank + 1) if rank <= 50 else 0.0)
            
            # AUC
            valid_negatives = (scores > -float('inf')).sum().item() - 1
            if valid_negatives > 0:
                auc = 1.0 - (rank - 1) / valid_negatives
                aucs.append(auc)
            
            processed += 1
            
        metrics = {
            "Recall@10": np.mean(recalls_at_10),
            "Recall@50": np.mean(recalls_at_50),
            "MRR": np.mean(mrrs),
            "NDCG@10": np.mean(ndcg_at_10),
            "NDCG@50": np.mean(ndcg_at_50),
            "AUC": np.mean(aucs)
        }
        
        logging.info(f"Final Results ({mode}):")
        for k, v in metrics.items():
            logging.info(f"{k}: {v:.4f}")
            
        return metrics

def run_benchmark_suite(args):
    """Runs evaluation for all defined models and saves a combined CSV."""
    
    # Define models to evaluate
    # Format: (Friendly Name, Model Type, Checkpoint Path/ID)
    models = [
        ("BLaIR-Base", "blair_base", "hyp1231/blair-roberta-base"),
        ("BLaIR-Large", "blair_large", "hyp1231/blair-roberta-large"),
        ("RoBERTa-CLIP-Unfrozen", "blair_clip", "checkpoints/roberta-clip-unfrozen"),
        ("RoBERTa-CLIP-Frozen", "blair_clip", "checkpoints/roberta-clip-frozen"),
        ("BLaIR-CLIP-Unfrozen", "blair_clip", "checkpoints/blair-clip-unfrozen"),
        ("BLaIR-CLIP-Frozen", "blair_clip", "checkpoints/blair-clip-frozen"),
    ]
    
    results_file = "evaluation_results.csv"
    
    splitter = CustomDataSplitter(data_dir=args.data_dir, meta_file=args.meta_file)
    splitter.load_data()
    
    all_results = []
    
    for friendly_name, model_type, checkpoint_path in models:
        # Check if checkpoint exists (for local paths)
        if "checkpoints/" in checkpoint_path:
            if not os.path.exists(checkpoint_path):
                logging.warning(f"Checkpoint not found: {checkpoint_path}. Skipping.")
                continue
            # Resolve actual checkpoint subfolder
            if not os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
                subdirs = [d for d in os.listdir(checkpoint_path) if d.startswith("checkpoint-")]
                if subdirs:
                    subdirs.sort(key=lambda x: int(x.split('-')[1]))
                    checkpoint_path = os.path.join(checkpoint_path, subdirs[-1])
                else:
                    logging.warning(f"No checkpoint subdirs found in {checkpoint_path}. Skipping.")
                    continue
        
        logging.info(f"=== Benchmarking {friendly_name} ===")
        
        evaluator = BlairEvaluator(checkpoint_path, splitter, batch_size=args.batch_size, device=args.device, model_type=model_type)
        evaluator.load_model()
        evaluator.encode_all_items(cache_file=f"item_embeddings_dict_{friendly_name}.pt", force_refresh=args.no_cache)
        
        modes = ['text', 'image', 'combined']
        for mode in modes:
            metrics = evaluator.evaluate(limit=args.limit, mode=mode)
            if metrics:
                row = {
                    "Model": friendly_name,
                    "Type": model_type,
                    "Mode": mode,
                    **metrics
                }
                all_results.append(row)
                
    # Write to CSV
    if all_results:
        keys = all_results[0].keys()
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        logging.info(f"Saved benchmark results to {results_file}")
    else:
        logging.warning("No results collected.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate BLaIR-CLIP model on sequential recommendation")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/blair-clip-base_new", 
                        help="Path to the model checkpoint directory (for single run)")
    parser.add_argument("--data_dir", type=str, default="splits/with_images",
                        help="Path to the directory containing split CSVs")
    parser.add_argument("--meta_file", type=str, default="./clean_review_meta_with_images.tsv",
                        help="Path to the metadata TSV file")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for item encoding")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test users for quick debugging")
    parser.add_argument("--cache_file", type=str, default="item_embeddings_dict.pt", help="Path to cache item embeddings (single run)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on")
    parser.add_argument("--model_type", type=str, default="blair_clip", 
                        choices=["blair_clip", "blair_base", "blair_large"],
                        help="Model type to evaluate (single run)")
    parser.add_argument("--run_all", action="store_true", help="Run full benchmark suite across all known models")
    parser.add_argument("--no-cache", action="store_true", help="Force re-encoding of items (ignore cache)")
    
    args = parser.parse_args()
    
    if args.run_all:
        run_benchmark_suite(args)
    else:
        # Legacy Single Run Mode
        checkpoint_dir = args.checkpoint_dir
        if args.model_type == 'blair_clip':
            if not os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
                try:
                    if os.path.exists(checkpoint_dir):
                        subdirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                        if subdirs:
                            subdirs.sort(key=lambda x: int(x.split('-')[1]))
                            checkpoint_dir = os.path.join(checkpoint_dir, subdirs[-1])
                            logging.info(f"Found checkpoint: {checkpoint_dir}")
                except FileNotFoundError:
                    pass
        else:
            if args.checkpoint_dir == "checkpoints/blair-clip-base_new":
                checkpoint_dir = "hyp1231/blair-roberta-base" if args.model_type == 'blair_base' else "hyp1231/blair-roberta-large"
        
        splitter = CustomDataSplitter(data_dir=args.data_dir, meta_file=args.meta_file)
        splitter.load_data()
        
        evaluator = BlairEvaluator(checkpoint_dir, splitter, batch_size=args.batch_size, device=args.device, model_type=args.model_type)
        evaluator.load_model()
        evaluator.encode_all_items(cache_file=args.cache_file, force_refresh=args.no_cache)
        
        modes = ['text', 'image', 'combined']
        for mode in modes:
            print(f"\n{'='*40}")
            print(f"Running Evaluation for Mode: {mode}")
            print(f"{'='*40}\n")
            evaluator.evaluate(limit=args.limit, mode=mode)

if __name__ == "__main__":
    main()
