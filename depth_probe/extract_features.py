import sys
sys.path.append('..')  # so we can import from the RayZer root

import torch
import torch.distributed

# Mock torch.distributed for single-GPU/CPU non-distributed run
if not torch.distributed.is_initialized():
    torch.distributed.is_initialized = lambda: True
    torch.distributed.get_rank = lambda: 0
    torch.distributed.get_world_size = lambda: 1
    torch.distributed.barrier = lambda: None

import numpy as np
from omegaconf import OmegaConf
from easydict import EasyDict as edict
from einops import rearrange
from torch.utils.data import DataLoader

from model.rayzer import RayZer
from dataset_nyu_depth import NYUDepthDataset


# ── Config ──────────────────────────────────────────────
CONFIG_PATH = '../configs/rayzer_dl3dv.yaml'
CHECKPOINT_PATH = '../model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt'
DATASET_PATH = 'nyu_depth_v2_labeled.mat'
OUTPUT_PATH = 'cached_features.pt'
BATCH_SIZE = 16       
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VAL_SPLIT = 0.2


def load_config(config_path):
    """Load yaml config and convert to edict (same as setup.py)."""
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = edict(config)
    return config

def load_model(config, checkpoint_path, device):
    """Instantiate RayZer, load checkpoint, freeze everything."""
    # 1. Instantiate RayZer(config)
    model = RayZer(config)
    # 2. Load checkpoint (watch for 'module.' prefix from DDP)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']  # checkpoint is a dict with 'model' key
    cleaned = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    # 3. model.eval(), freeze all params
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # 4. Move to device
    model = model.to(device)
    return model


def extract_encoder_tokens(model, images):
    """
    Run the encoder portion of RayZer on a batch of single images.
    
    Args:
        model: frozen RayZer model
        images: [B, 3, 256, 256] in range [-1, 1]
    
    Returns:
        img_tokens: [B, 256, 768] post-encoder image tokens
    """
    b = images.shape[0]
    v = 24  # use v=24 and duplicate images to get cached features
    h, w = 256, 256

    # NOTE: Your dataset already outputs [-1, 1]. RayZer's forward() does
    # image * 2.0 - 1.0 because it expects [0, 1]. Don't double-normalize!
    # Just reshape to [B, V, C, H, W] = [B, 1, 3, 256, 256].

    # TODO: Step A — Reshape for RayZer's multi-view format
    # Repeat the image v times
    image_all = images.unsqueeze(1).repeat(1, v, 1, 1, 1)

    # TODO: Step B — Tokenize: model.image_tokenizer(...)
    img_tokens = model.image_tokenizer(image_all)

    # TODO: Step C — Add positional embedding: model.add_sptial_temporal_pe(...)
    if model.use_pe_embedding_layer:
        img_tokens = model.add_sptial_temporal_pe(img_tokens, b, v, h, w)

    # TODO: Step D — Rearrange: rearrange(img_tokens, '(b v) n d -> b (v n) d', ...)
    img_tokens = rearrange(img_tokens, '(b v) n d -> b (v n) d', b=b, v=v)  # [B, 256, 768]

    # TODO: Step E — Get camera tokens: model.get_camera_tokens(b, v=1)
    cam_tokens = model.get_camera_tokens(b, v)  

    # TODO: Step F — Concat cam + img, run encoder, split out img_tokens
    n_cam = cam_tokens.shape[1] // v  # = 1
    all_tokens = torch.cat([cam_tokens, img_tokens], dim=1)  # [B, 257, 768]
    all_tokens = model.run_encoder(all_tokens)
    _, img_tokens = all_tokens.split([v * n_cam, v * 256], dim=1)

    # We copied the image 24 times. To keep the output shape identical
    # to single-view [B, 256, 768], we slice out the first view's features.
    img_tokens = img_tokens[:, :256, :]

    return img_tokens


def create_split_indices(dataset_size, val_split, seed=42):
    """Create random train/val index split."""
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=g)
    val_size = int(dataset_size * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def main():
    # ── Setup ───────────────────────────────────────────
    config = load_config(CONFIG_PATH)
    model = load_model(config, CHECKPOINT_PATH, DEVICE)
    dataset = NYUDepthDataset(mat_path=DATASET_PATH)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,          # preserve ordering for index mapping
        num_workers=4,
        pin_memory=True,
    )

    # ── Extract features ────────────────────────────────
    all_features = []
    all_depths = []

    print("Extracting features from RayZer encoder...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(DEVICE)   # [B, 3, 256, 256]
            depths = batch['depth']               # [B, 1, 16, 16] — keep on CPU

            # Extract tokens with autocast if on CUDA (needed for xformers on Turing GPUs)
            if 'cuda' in str(DEVICE):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    tokens = extract_encoder_tokens(model, images)  # [B, 256, 768]
            else:
                tokens = extract_encoder_tokens(model, images)

            all_features.append(tokens.cpu().float())
            all_depths.append(depths)

    # ── Concatenate ─────────────────────────────────────
    all_features = torch.cat(all_features, dim=0)  # [1449, 256, 768]
    all_depths = torch.cat(all_depths, dim=0)       # [1449, 1, 16, 16]

    # ── Split ───────────────────────────────────────────
    train_indices, val_indices = create_split_indices(len(dataset), VAL_SPLIT)

    # ── Save ────────────────────────────────────────────
    torch.save({
        'features': all_features,
        'depths': all_depths,
        'train_indices': train_indices,
        'val_indices': val_indices,
    }, OUTPUT_PATH)

    print(f"Saved to {OUTPUT_PATH}")
    print(f"Features: {all_features.shape}")
    print(f"Depths: {all_depths.shape}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")


if __name__ == '__main__':
    main()
