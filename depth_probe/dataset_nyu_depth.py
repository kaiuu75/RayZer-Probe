import h5py
import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset

class LogQuantileNormalize:
    """Custom transform: normalizes depth to [-1, 1] using log + quantiles.

    Formula: d̃ = ((log(d) - log(d2)) / (log(d98) - log(d2)) - 0.5) * 2
    """
    def __init__(self, d2, d98):
        self.log_d2 = torch.log(d2.clamp(min=1e-6))
        self.log_d98 = torch.log(d98.clamp(min=1e-6))

    def __call__(self, depth):
        log_depth = torch.log(depth.clamp(min=1e-6))
        normalized = ((log_depth - self.log_d2) / (self.log_d98 - self.log_d2) - 0.5) * 2
        return normalized.clamp(-1, 1)

class NYUDepthDataset(Dataset):
    def __init__(self, mat_path):
        f = h5py.File(mat_path, 'r')
        self.images = np.array(f['images']).transpose(0, 3, 2, 1) 
        self.depths = np.array(f['depths']).transpose(0, 2, 1)
        f.close()

        #calculate quantiles
        all_depths = self.depths.flatten() 
        self.d2 = torch.tensor(np.percentile(all_depths, 2), dtype=torch.float32)
        self.d98 = torch.tensor(np.percentile(all_depths, 98), dtype=torch.float32)

        #image transform
        self.image_transforms = v2.Compose([
            v2.ToImage(),
            v2.CenterCrop(256),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        #depth transform
        self.depth_transforms = v2.Compose([
            v2.ToImage(),
            v2.CenterCrop(256),
            v2.ToDtype(torch.float32),
            LogQuantileNormalize(self.d2, self.d98),
            v2.Resize(16)
        ])


    def __len__(self):
        return len(self.images) 

    def __getitem__(self, idx):
        image = self.image_transforms(self.images[idx])
        depth = self.depth_transforms(self.depths[idx])

        return {'image': image, 'depth': depth}