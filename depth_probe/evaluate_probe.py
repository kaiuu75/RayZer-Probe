import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import json

from load_features import FeatureDataset
from probes import get_probe, PROBE_CHOICES

# ── Config ──────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            tokens = batch['tokens'].to(DEVICE)
            targets = batch['depth'].to(DEVICE)
            predictions = model(tokens)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * tokens.size(0)
    return total_loss / len(loader.dataset)

def delta1_accuracy(pred, gt, threshold=1.25):
    """Compute δ₁ accuracy: % of pixels where max(pred/gt, gt/pred) < threshold.
    
    Both pred and gt are in normalized [-1, 1] space. We exponentiate to map
    them to positive values before computing the ratio, preserving the
    relative ordering from the log-quantile normalization.
    """
    # Map from [-1, 1] to positive depth via exp()
    pred_pos = torch.exp(pred)
    gt_pos = torch.exp(gt)
    
    ratio = torch.max(pred_pos / gt_pos, gt_pos / pred_pos)
    return (ratio < threshold).float().mean().item()

def evaluate_delta1(model, loader, threshold=1.25):
    """Compute δ₁ accuracy over an entire dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            tokens = batch['tokens'].to(DEVICE)
            targets = batch['depth'].to(DEVICE)
            predictions = model(tokens)
            
            pred_pos = torch.exp(predictions)
            gt_pos = torch.exp(targets)
            ratio = torch.max(pred_pos / gt_pos, gt_pos / pred_pos)
            
            correct += (ratio < threshold).sum().item()
            total += ratio.numel()
    return correct / total

def append_block_results(output_dir, block, probe_type, train_mse, val_mse, train_delta1, val_delta1):
    """Append metrics for one block to the shared block_results.json file."""
    results_path = os.path.join(output_dir, 'block_results.json')
    
    # Load existing results or start fresh
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = []
    
    # Remove any existing entry for this block+probe combo (in case of re-runs)
    results = [r for r in results if not (r.get('block') == block and r['probe'] == probe_type)]
    
    results.append({
        'block': block,
        'probe': probe_type,
        'train_mse': train_mse,
        'val_mse': val_mse,
        'train_delta1': train_delta1,
        'val_delta1': val_delta1,
    })
    
    # Sort by block index for readability (pre_encoder is handled differently)
    def sort_key(r):
        b = r.get('block', '')
        if b == 'pre_encoder': return -1
        try: return int(b.split('_')[1])
        except: return 999
        
    results.sort(key=sort_key)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  --> Appended results to {results_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='.', help='Directory containing checkpoint and for saving outputs')
    parser.add_argument('--probe', type=str, default='linear', choices=PROBE_CHOICES,
                        help='Probe type to evaluate (default: linear)')
    parser.add_argument('--block', type=str, required=True,
                        help='Which block to evaluate (e.g., pre_encoder, block_0).')
    parser.add_argument('--cached-features', type=str, default='cached_features.pt',
                        help='Path to cached features file.')
    args = parser.parse_args()
    
    cached_path = args.cached_features
    
    checkpoint_name = f'best_probe_{args.block}.pt'
    checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
    
    print(f"Using device: {DEVICE}")
    print(f"Probe type: {args.probe}")
    print(f"Block: {args.block}")
    print(f"Output directory: {args.output_dir}")
    
    # 1. Load data
    train_dataset = FeatureDataset(cached_path=cached_path, split='train', block=args.block)
    val_dataset = FeatureDataset(cached_path=cached_path, split='val', block=args.block)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 2. Setup model and loss
    criterion = nn.MSELoss()
    
    trained_model = get_probe(args.probe, input_dim=768).to(DEVICE)
    trained_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    
    # 3. Compute metrics
    train_mse = evaluate(trained_model, train_loader, criterion)
    val_mse = evaluate(trained_model, val_loader, criterion)
    
    train_delta1 = evaluate_delta1(trained_model, train_loader)
    val_delta1 = evaluate_delta1(trained_model, val_loader)
    
    print("\n" + "="*40)
    print(f"    DEPTH PROBE EVALUATION [{args.block}]")
    print("="*40)
    print(f"  Train MSE:       {train_mse:.6f}")
    print(f"  Val MSE:         {val_mse:.6f}")
    print(f"  Train δ₁ (α=1):  {train_delta1:.4f}  ({train_delta1*100:.1f}%)")
    print(f"  Val δ₁ (α=1):    {val_delta1:.4f}  ({val_delta1*100:.1f}%)")
    print("="*40 + "\n")
    
    # 4. Write results to JSON (for blockwise comparison plot)
    append_block_results(args.output_dir, args.block, args.probe,
                         train_mse, val_mse, train_delta1, val_delta1)
    
    # 5. Generate visualization for specific samples
    trained_model.eval()
    
    sample_indices = [100, 500, 900, 1300]
    cached_data = torch.load(cached_path, weights_only=True)
    
    # Get features directly using global indices
    tokens = cached_data[args.block][sample_indices].to(DEVICE)
    targets = cached_data['depths'][sample_indices].numpy()
    
    with torch.no_grad():
        predictions = trained_model(tokens).cpu().numpy()
        
    from dataset_nyu_depth import NYUDepthDataset
    nyu = NYUDepthDataset(mat_path='nyu_depth_v2_labeled.mat')
    
    # Plotting: 4 rows x 3 cols (RGB | GT Depth | Predicted Depth)
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    axes[0, 0].set_title("RGB Image", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("GT Depth", fontsize=12, fontweight='bold')
    axes[0, 2].set_title(f"Predicted Depth [{args.block}]", fontsize=12, fontweight='bold')

    vmin, vmax = -1, 1

    for i in range(len(sample_indices)):
        idx = sample_indices[i]
        # RGB image (undo normalization: img * 0.5 + 0.5 -> [0, 1])
        nyu_sample = nyu[idx]
        rgb = nyu_sample['image'].permute(1, 2, 0).numpy() * 0.5 + 0.5
        rgb = np.clip(rgb, 0, 1)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_ylabel(f"idx {idx}", fontsize=10)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # GT Depth
        im_gt = axes[i, 1].imshow(targets[i, 0], cmap='inferno', vmin=vmin, vmax=vmax)
        axes[i, 1].axis('off')

        # Predicted Depth (with colorbar)
        im_pred = axes[i, 2].imshow(predictions[i, 0], cmap='inferno', vmin=vmin, vmax=vmax)
        axes[i, 2].axis('off')
        fig.colorbar(im_pred, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plot_name = f'probe_visualization_{args.block}.png'
    plot_path = os.path.join(args.output_dir, plot_name)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved visualization grid to {plot_path}")

if __name__ == '__main__':
    main()

