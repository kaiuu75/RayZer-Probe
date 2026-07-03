import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """Dataset class for loading pre-extracted features and depth maps.
    
    Args:
        cached_path: Path to cached features file.
        split: 'train' or 'val'.
        block: Which block's features to load. A string like 'pre_encoder', 'block_0',
               'block_5'.
    """
    def __init__(self, cached_path='cached_features.pt', split='train', block='block_7'):
        data = torch.load(cached_path, weights_only=True)
        
        if block not in data:
            available = [k for k in data.keys() if k.startswith('block_') or k == 'pre_encoder']
            raise ValueError(f"Block '{block}' not found. Available: {available}")
        
        features = data[block]
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