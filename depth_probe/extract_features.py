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

import argparse
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
    Run the encoder portion of RayZer on a batch of single images,
    capturing intermediate representations after each encoder layer.
    
    Args:
        model: frozen RayZer model
        images: [B, 3, 256, 256] in range [-1, 1]
    
    Returns:
        dict: block_name -> [B, 256, 768] image tokens
              pre_encoder = after tokenize + PE, before any block
              block_0..block_N = after encoder block 0..N
    """
    b = images.shape[0]
    v = 24
    h, w = 256, 256

    # Steps A–E: identical to extract_encoder_tokens
    image_all = images.unsqueeze(1).repeat(1, v, 1, 1, 1)
    img_tokens = model.image_tokenizer(image_all)
    if model.use_pe_embedding_layer:
        img_tokens = model.add_sptial_temporal_pe(img_tokens, b, v, h, w)
    img_tokens = rearrange(img_tokens, '(b v) n d -> b (v n) d', b=b, v=v)
    cam_tokens = model.get_camera_tokens(b, v)

    # Step F — Run encoder blocks one by one, capturing intermediates
    n_cam = cam_tokens.shape[1] // v  # = 1
    all_tokens = torch.cat([cam_tokens, img_tokens], dim=1)

    block_outputs = {}

    # pre_encoder: embedding (after tokenize + PE, before any block)
    _, pre_img = all_tokens.split([v * n_cam, v * 256], dim=1)
    block_outputs['pre_encoder'] = pre_img[:, :256, :].clone()

    # block_0..block_N: after each encoder block
    for i, block in enumerate(model.transformer_encoder):
        all_tokens = block(all_tokens)
        _, img_out = all_tokens.split([v * n_cam, v * 256], dim=1)
        block_outputs[f'block_{i}'] = img_out[:, :256, :].clone()

    return block_outputs


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

    # ── Blockwise extraction ────────────────────────
    num_blocks = len(model.transformer_encoder)
    block_names = ['pre_encoder'] + [f'block_{i}' for i in range(num_blocks)]
    all_block_features = {name: [] for name in block_names}
    all_depths = []

    print(f"Extracting blockwise features from RayZer encoder ({num_blocks} blocks + pre-encoder)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(DEVICE)
            depths = batch['depth']

            if 'cuda' in str(DEVICE):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    block_outputs = extract_encoder_tokens(model, images)
            else:
                block_outputs = extract_encoder_tokens(model, images)

            for name in block_names:
                all_block_features[name].append(block_outputs[name].cpu().float())
            all_depths.append(depths)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} images...")

    # ── Concatenate ─────────────────────────────────
    for name in block_names:
        all_block_features[name] = torch.cat(all_block_features[name], dim=0)
    all_depths = torch.cat(all_depths, dim=0)

    # ── Split ───────────────────────────────────────
    train_indices, val_indices = create_split_indices(len(dataset), VAL_SPLIT)

    # ── Save ────────────────────────────────────────
    save_dict = {
        'depths': all_depths,
        'train_indices': train_indices,
        'val_indices': val_indices,
    }
    save_dict.update(all_block_features)

    torch.save(save_dict, OUTPUT_PATH)

    print(f"Saved to {OUTPUT_PATH}")
    for name in block_names:
        print(f"  {name}: {all_block_features[name].shape}")
    print(f"Depths: {all_depths.shape}")
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")


if __name__ == '__main__':
    main()
