# Probing for Monocular Depth Estimation in Self-Supervised View Synthesis Models

This repository is a fork of [RayZer](https://github.com/hwjiang1510/RayZer) (Jiang et al., 2025), adapted for **probing experiments** that investigate whether the transformer representations learned by RayZer encode meaningful depth information. Specifically, we extract intermediate features from RayZer's frozen encoder and train lightweight probes to predict monocular depth, evaluated on the [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) benchmark.

> **Note**: This repository is part of a thesis project and is not affiliated with the original RayZer authors or Adobe.

---

## Motivation

RayZer is a self-supervised model that learns to synthesize novel views without explicit 3D supervision. A natural question arises: *do the internal representations of such a model implicitly capture 3D scene geometry, particularly depth?* This repository provides the experimental infrastructure to answer that question via probing—a standard protocol for evaluating learned representations (Alain & Bengio, 2017).

---

## Method

1. **Feature Extraction** — We forward-pass images through RayZer's frozen transformer encoder and extract token-level activations from selected layers.
2. **Probing** — A lightweight probe is trained on top of these frozen features to predict per-pixel depth maps. Two probe architectures are supported:
   - **Linear** — A single linear layer (`768 → 1`). Tests what is linearly decodable from the representation.
   - **MLP** — A 3-layer MLP with GELU activations (`768 → 768 → 768 → 1`). Tests what is non-linearly accessible.
3. **Evaluation** — Predictions are compared against ground-truth depth from NYU Depth V2 using standard metrics (MSE, δ < 1.25, etc.).

---

## Repository Structure

```
depth_probe/
├── probes/                  # Probe architectures
│   ├── __init__.py          # Factory function get_probe() and PROBE_CHOICES
│   ├── linear_probe.py      # Single linear layer probe
│   └── mlp_probe.py         # 3-layer MLP probe with GELU
├── extract_features.py      # Extract and cache token features from frozen RayZer
├── load_features.py         # Dataset class for cached features
├── train_probe.py           # Training script (--probe linear|mlp)
├── evaluate_probe.py        # Evaluation and visualization (--probe linear|mlp)
├── dataset_nyu_depth.py     # NYU Depth V2 dataset loader
├── start_probe.sh           # SLURM job script (set PROBE_TYPE=linear|mlp)
└── logs/                    # Timestamped run directories with output logs
```

---

## Setup

### Environment

```bash
conda create -n rayzer python=3.11
conda activate rayzer
pip install -r requirements.txt
```

GPU compute capability ≥ 8.0 is required (see [CUDA GPUs](https://developer.nvidia.com/cuda-gpus#compute)) due to the use of [xformers](https://github.com/facebookresearch/xformers) memory-efficient attention.

### Data

**NYU Depth V2** — Download the dataset from the [official page](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) or via the preprocessed HDF5 version. Place the data file in the project root or update the path in `data/dataset_nyu_depth.py`.

**RayZer Checkpoints** — Pretrained RayZer weights are available from the original repository:

| Data   | Model                      | Link |
|--------|----------------------------|------|
| DL3DV  | RayZer-8-12-12-100K       | [Hugging Face](https://huggingface.co/hwjiang/RayZer/resolve/main/rayzer_dl3dv_8_12_12_96k.pt?download=true) |

### Usage

Train and evaluate a probe by selecting the probe type:

```bash
cd depth_probe

# Linear probe (default)
python train_probe.py --output-dir runs/linear --probe linear
python evaluate_probe.py --output-dir runs/linear --probe linear

# MLP probe
python train_probe.py --output-dir runs/mlp --probe mlp
python evaluate_probe.py --output-dir runs/mlp --probe mlp
```

Or via the SLURM script:

```bash
# Default (linear)
sbatch start_probe.sh

# MLP probe
PROBE_TYPE=mlp sbatch start_probe.sh
```

---

## Acknowledgements

This work builds directly on the following projects and datasets. We gratefully acknowledge the original authors:

### RayZer

This codebase is forked from the official RayZer repository. The model architecture, training pipeline, and pretrained checkpoints are the work of Jiang et al.

> Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, and Georgios Pavlakos.
> *"RayZer: A Self-supervised Large View Synthesis Model."*
> ICCV 2025 (Oral). arXiv: [2505.00702](https://arxiv.org/abs/2505.00702).
>
> Project page: [https://hwjiang1510.github.io/RayZer/](https://hwjiang1510.github.io/RayZer/)

The RayZer codebase is itself developed based on [LVSM](https://github.com/Haian-Jin/LVSM).

### NYU Depth V2

We use the NYU Depth V2 dataset for depth evaluation.

> Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus.
> *"Indoor Segmentation and Support Inference from RGBD Images."*
> ECCV 2012.
>
> Dataset: [https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

### Linear Probing Methodology

The linear probing protocol follows the framework established by:

> Guillaume Alain and Yoshua Bengio.
> *"Understanding intermediate layers using linear classifier probes."*
> ICLR 2017 Workshop.

---

## Citations

If you use this repository, please cite the original RayZer paper and the NYU Depth V2 dataset:

```bibtex
@article{jiang2025rayzer,
  title     = {RayZer: A Self-supervised Large View Synthesis Model},
  author    = {Jiang, Hanwen and Tan, Hao and Wang, Peng and Jin, Haian and
               Zhao, Yue and Bi, Sai and Zhang, Kai and Luan, Fujun and
               Sunkavalli, Kalyan and Huang, Qixing and Pavlakos, Georgios},
  journal   = {arXiv preprint arXiv:2505.00702},
  year      = {2025}
}

@inproceedings{silberman2012indoor,
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  author    = {Silberman, Nathan and Hoiem, Derek and Kohli, Pushmeet and
               Fergus, Rob},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2012}
}
```

---

## License

This repository inherits the [CC BY-NC-SA 4.0](LICENSE.md) license from the original RayZer codebase.
