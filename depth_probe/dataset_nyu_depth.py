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



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    dataset = NYUDepthDataset(mat_path='nyu_depth_v2_labeled.mat')
    print(f"Dataset size: {len(dataset)}")
    print(f"d2 quantile: {dataset.d2:.4f}")
    print(f"d98 quantile: {dataset.d98:.4f}")

    sample = dataset[500]
    image = sample['image']
    depth = sample['depth']

    print(f"\nImage shape: {image.shape}")
    print(f"Image range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")

    # Visualize 4 samples
    sample_indices = [100, 500, 900, 1300]
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    axes[0, 0].set_title("RGB Image", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("Depth (meters)", fontsize=12, fontweight='bold')
    axes[0, 2].set_title("Depth (16x16 normalized)", fontsize=12, fontweight='bold')

    for i, idx in enumerate(sample_indices):
        s = dataset[idx]

        # RGB (undo normalization)
        rgb = s['image'].permute(1, 2, 0).numpy() * 0.5 + 0.5
        rgb = np.clip(rgb, 0, 1)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_ylabel(f"idx {idx}", fontsize=10)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Raw depth (full resolution, in meters)
        raw_depth = dataset.depths[idx]
        im_raw = axes[i, 1].imshow(raw_depth, cmap='inferno')
        axes[i, 1].axis('off')
        fig.colorbar(im_raw, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Resized normalized depth (16x16, [-1, 1])
        im_norm = axes[i, 2].imshow(s['depth'][0], cmap='inferno', vmin=-1, vmax=1)
        axes[i, 2].axis('off')
        fig.colorbar(im_norm, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('nyu_samples.png', dpi=150)
    print("\nSaved visualization to nyu_samples.png")
