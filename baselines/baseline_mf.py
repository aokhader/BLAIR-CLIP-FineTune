import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# Add parent directory to path to import baseline_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline_utils import DataSplitter, Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(BPRMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user, item):
        u_emb = self.user_embedding(user)
        i_emb = self.item_embedding(item)
        return (u_emb * i_emb).sum(dim=1)

class BPRDataset(Dataset):
    def __init__(self, train_interactions, num_items, num_samples=1):
        self.train_interactions = train_interactions
        self.num_items = num_items
        self.num_samples = num_samples
        
        self.users = list(train_interactions.keys())
        self.user_items = {u: set(items) for u, items in train_interactions.items()}

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_items = list(self.user_items[user])
        
        if not pos_items:
            # Should not happen if filtered correctly
            pos_item = 0
        else:
            pos_item = random.choice(pos_items)
            
        # Negative sampling
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_items[user]:
                break
                
        return torch.tensor(user, dtype=torch.long), torch.tensor(pos_item, dtype=torch.long), torch.tensor(neg_item, dtype=torch.long)

class MFBaseline:
    def __init__(self, data_dir='.', meta_file='meta_Appliances.jsonl', reviews_file='Appliances.jsonl', 
                 embedding_dim=64, batch_size=1024, lr=0.001, epochs=10, device='cpu'):
        self.splitter = DataSplitter(data_dir, meta_file, reviews_file)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
             self.device = 'mps'
        
        self.model = None
        self.user_to_index = {}
        self.index_to_user = {}

    def run(self):
        logging.info(f"Starting Matrix Factorization Baseline (BPR) on {self.device}")
        
        # 1. Load and Split Data
        self.splitter.load_data()
        self.splitter.preprocess()
        
        # Map users to indices (0 to N-1)
        # We need to include ALL users from train and test to ensure consistent mapping
        # Actually, DataSplitter gives us train_interactions and test_interactions.
        # We need a global user mapping.
        
        all_users = set(self.splitter.train_interactions.keys())
        for u, _ in self.splitter.test_interactions:
            all_users.add(u)
            
        self.user_to_index = {u: i for i, u in enumerate(all_users)}
        self.index_to_user = {i: u for u, i in self.user_to_index.items()}
        
        num_users = len(all_users)
        num_items = len(self.splitter.all_asins)
        
        logging.info(f"Num Users: {num_users}, Num Items: {num_items}")
        
        # Convert interactions to indices
        train_interactions_idx = {}
        for u, items in self.splitter.train_interactions.items():
            u_idx = self.user_to_index[u]
            i_idxs = [self.splitter.asin_to_index[i] for i in items if i in self.splitter.asin_to_index]
            if i_idxs:
                train_interactions_idx[u_idx] = i_idxs
                
        # 2. Initialize Model
        self.model = BPRMF(num_users, num_items, self.embedding_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # 3. Training Loop
        logging.info("Training...")
        dataset = BPRDataset(train_interactions_idx, num_items)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for user, pos_item, neg_item in dataloader:
                user = user.to(self.device)
                pos_item = pos_item.to(self.device)
                neg_item = neg_item.to(self.device)
                
                optimizer.zero_grad()
                
                pos_scores = self.model(user, pos_item)
                neg_scores = self.model(user, neg_item)
                
                loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            logging.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")

        # 4. Evaluation
        self.model.eval()
        
        # Pre-compute item embeddings for fast scoring
        # item_emb: (N_items, Dim)
        item_emb = self.model.item_embedding.weight.detach().cpu().numpy()
        
        # We need a wrapper for Evaluator
        # Evaluator expects score_func(user_id) -> scores
        # But Evaluator uses original user_ids (strings).
        
        # Also Evaluator expects train_interactions with original IDs.
        # We passed self.splitter.train_interactions which has original IDs. Correct.
        
        evaluator = Evaluator(
            self.splitter.train_interactions,
            self.splitter.test_interactions,
            self.splitter.asin_to_index,
            num_items
        )
        
        def score_func(user_id):
            if user_id not in self.user_to_index:
                return np.full(num_items, -np.inf)
            
            u_idx = self.user_to_index[user_id]
            
            # Get user embedding
            # We can use the model or just lookup since we are in eval mode
            # But model is on device.
            
            with torch.no_grad():
                u_tensor = torch.tensor([u_idx], device=self.device)
                u_emb = self.model.user_embedding(u_tensor).cpu().numpy() # (1, Dim)
            
            # Score = u_emb @ item_emb.T
            scores = u_emb.dot(item_emb.T).flatten()
            return scores

        return evaluator.evaluate(score_func)

if __name__ == "__main__":
    # Check if data files exist in current dir, else try parent
    if os.path.exists('meta_Appliances.jsonl'):
        data_dir = '.'
    elif os.path.exists('../meta_Appliances.jsonl'):
        data_dir = '..'
    else:
        data_dir = '.' 
        
    model = MFBaseline(data_dir=data_dir, epochs=5) # 5 epochs for speed, can increase
    model.run()
