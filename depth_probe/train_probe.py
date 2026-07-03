import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os

from load_features import FeatureDataset
from probes import get_probe, PROBE_CHOICES

# ── Config ──────────────────────────────────────────────
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save checkpoint')
    parser.add_argument('--probe', type=str, default='linear', choices=PROBE_CHOICES,
                        help='Probe type to train (default: linear)')
    parser.add_argument('--block', type=str, required=True,
                        help='Which block to probe (e.g., pre_encoder, block_0).')
    parser.add_argument('--cached-features', type=str, default='cached_features.pt',
                        help='Path to cached features file. Defaults to cached_features.pt')
    args = parser.parse_args()
    
    cached_path = args.cached_features
    
    # Determine checkpoint name (include block to avoid collisions)
    checkpoint_name = f'best_probe_{args.block}.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
    
    print(f"Using device: {DEVICE}")
    print(f"Probe type: {args.probe}")
    print(f"Block: {args.block}")
    print(f"Cached features: {cached_path}")
    print(f"Output directory: {args.output_dir}")
    
    # 1. Datasets and Dataloaders
    train_dataset = FeatureDataset(cached_path=cached_path, split='train', block=args.block)
    val_dataset = FeatureDataset(cached_path=cached_path, split='val', block=args.block)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 2. Model, Optimizer, Loss
    model = get_probe(args.probe, input_dim=768).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            tokens = batch['tokens'].to(DEVICE)  # [B, 256, 768]
            targets = batch['depth'].to(DEVICE)  # [B, 1, 16, 16]
            
            optimizer.zero_grad()
            predictions = model(tokens)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * tokens.size(0)
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(DEVICE)
                targets = batch['depth'].to(DEVICE)
                
                predictions = model(tokens)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * tokens.size(0)
                
        val_loss /= len(val_dataset)
        
        print(f"Epoch {epoch}/{EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best model to {checkpoint_path} with Val MSE: {val_loss:.6f}")

if __name__ == '__main__':
    main()
