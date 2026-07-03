"""
Plot per-block probing results: δ₁ accuracy and MSE across encoder blocks.

Reads block_results.json from the run directory and produces block_comparison.png.
"""
import json
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Run directory containing block_results.json')
    args = parser.parse_args()

    results_path = os.path.join(args.output_dir, 'block_results.json')
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run evaluate_probe.py with --block first.")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    if not results:
        print("No results found in block_results.json")
        return

    # Extract data (already sorted by block index)
    blocks = [r['block'] for r in results]
    
    def get_block_index(name):
        if name == 'pre_encoder': return -1
        return int(name.split('_')[1])
        
    block_indices = [get_block_index(b) for b in blocks]
    val_delta1 = [r['val_delta1'] * 100 for r in results]  # convert to %
    train_delta1 = [r['train_delta1'] * 100 for r in results]
    val_mse = [r['val_mse'] for r in results]
    train_mse = [r['train_mse'] for r in results]
    probe_type = results[0]['probe']

    # Find best block
    best_idx = np.argmax(val_delta1)
    best_block_idx_val = block_indices[best_idx]
    best_val_delta1 = val_delta1[best_idx]

    # ── Create figure ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f'Per-Block Depth Probing — RayZer Encoder ({probe_type} probe)',
                 fontsize=14, fontweight='bold', y=1.02)

    # ── Left plot: δ₁ accuracy ─────────────────────────
    ax1.plot(block_indices, val_delta1, 'o-', color='#2563eb', linewidth=2,
             markersize=8, label='Val δ₁', zorder=3)
    ax1.plot(block_indices, train_delta1, 's--', color='#93c5fd', linewidth=1.5,
             markersize=6, label='Train δ₁', alpha=0.7, zorder=2)

    # Highlight best block
    best_label_str = blocks[best_idx]
    ax1.scatter([best_block_idx_val], [best_val_delta1], s=200, color='#dc2626',
                zorder=4, marker='*', label=f'Best: {best_label_str} ({best_val_delta1:.1f}%)')

    # Block labels
    x_labels = [b for b in blocks]
    ax1.set_xticks(block_indices)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel('Encoder Block', fontsize=11)
    ax1.set_ylabel('δ₁ Accuracy (%)', fontsize=11)
    ax1.set_title('δ₁ Accuracy (↑ better)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    # ── Right plot: MSE ────────────────────────────────
    ax2.plot(block_indices, val_mse, 'o-', color='#dc2626', linewidth=2,
             markersize=8, label='Val MSE', zorder=3)
    ax2.plot(block_indices, train_mse, 's--', color='#fca5a5', linewidth=1.5,
             markersize=6, label='Train MSE', alpha=0.7, zorder=2)

    ax2.set_xticks(block_indices)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel('Encoder Block', fontsize=11)
    ax2.set_ylabel('MSE', fontsize=11)
    ax2.set_title('Mean Squared Error (↓ better)', fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'block_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved block comparison plot to {plot_path}")

    # ── Print summary table ────────────────────────────
    print("\n" + "="*60)
    print("  BLOCK COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Block':<12} {'Val δ₁ (%)':<14} {'Val MSE':<12} {'Train δ₁ (%)':<14} {'Train MSE'}")
    print("-"*60)
    for i, block in enumerate(blocks):
        marker = " ★" if i == best_idx else ""
        print(f"  {block:<12} {val_delta1[i]:<14.1f} {val_mse[i]:<12.6f} {train_delta1[i]:<14.1f} {train_mse[i]:.6f}{marker}")
    print("="*60)
    print(f"  Best block: {blocks[best_idx]} (Val δ₁ = {best_val_delta1:.1f}%)")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
