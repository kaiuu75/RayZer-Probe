import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """Dataset class for loading pre-extracted features and depth maps."""
    def __init__(self, cached_path='cached_features.pt', split='train'):
        data = torch.load(cached_path, weights_only=True)
        features = data['features']
        depths = data['depths']
        
        if split == 'train':
            indices = data['train_indices']
        elif split == 'val':
            indices = data['val_indices']
        else:
            raise ValueError(f"Invalid split: {split}")
            
        self.features = features[indices]
        self.depths = depths[indices]
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            'tokens': self.features[idx],  # [256, 768]
            'depth': self.depths[idx]      # [1, 16, 16]
        }


if __name__ == '__main__':
    CACHED_PATH = 'cached_features.pt'

    # 1. Load both splits
    train_ds = FeatureDataset(cached_path=CACHED_PATH, split='train')
    val_ds = FeatureDataset(cached_path=CACHED_PATH, split='val')
    print(f"✓ Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 2. Check no overlap between splits
    raw = torch.load(CACHED_PATH, weights_only=True)
    train_set = set(raw['train_indices'].tolist())
    val_set = set(raw['val_indices'].tolist())
    overlap = train_set & val_set
    assert len(overlap) == 0, f"Train/val overlap: {overlap}"
    assert len(train_set) + len(val_set) == len(raw['features']), "Indices don't cover full dataset"
    print(f"✓ No train/val overlap, indices cover full dataset")

    # 3. Shape checks on a sample
    sample = train_ds[0]
    assert sample['tokens'].shape == (256, 768), f"Bad token shape: {sample['tokens'].shape}"
    assert sample['depth'].shape == (1, 16, 16), f"Bad depth shape: {sample['depth'].shape}"
    print(f"✓ Token shape: {sample['tokens'].shape}, Depth shape: {sample['depth'].shape}")

    # 4. Depth value range
    all_depths = train_ds.depths
    print(f"✓ Depth range: [{all_depths.min():.4f}, {all_depths.max():.4f}] (expected ≈ [-1, 1])")

    print("\nAll checks passed!")
