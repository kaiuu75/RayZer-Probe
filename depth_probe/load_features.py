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