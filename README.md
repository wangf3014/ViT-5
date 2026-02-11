# ViT-5: Vision Transformers for the Mid-2020s

Official implementation of  
**ViT-5: Vision Transformers for the Mid-2020s**

📄 Paper: https://arxiv.org/abs/2602.08071  
🤗 Hugging Face: https://huggingface.co/FengWang3211/ViT-5  

---

## Overview

ViT-5 modernizes the canonical Vision Transformer architecture while preserving the clean Attention–FFN backbone design.

Rather than introducing a new paradigm, ViT-5 systematically upgrades core components of ViT using insights accumulated over the past several years in large-scale vision modeling.

It serves as:

- A strong ImageNet classification backbone  
- A scalable Transformer foundation  
- A drop-in upgrade for modern vision pipelines  
- A competitive backbone for generative modeling  

---

## Architecture

<p align="center">
  <img src="assets/fig1.png" width="48%" />
  <img src="assets/fig4.png" width="48%" />
</p>

ViT-5 keeps the standard Transformer encoder:

Patch Embedding → [Attention → FFN] × L → Head

but modernizes:

- Normalization  
- Positional encoding  
- Optimization recipe  
- Stabilization techniques  
- Training strategy  

Full details are described in the paper.

---

## Generative Modeling Impact

<p align="center">
  <img src="assets/fig6.png" width="75%" />
</p>

ViT-5 also improves performance when used as a backbone in diffusion-style generative frameworks.

---

# Results & Checkpoints

| Model | Input Resolution | Params | Top-1 (ImageNet-1K) | HF Link |
|-------|------------------|--------|---------------------|---------|
| ViT-5-Small | 224 | 22M  | 82.2% | [Download](https://huggingface.co/FengWang3211/ViT-5/blob/main/vit5_small_patch16_224.pth) |
| ViT-5-Base  | 224 | 87M  | 84.2% | [Download](https://huggingface.co/FengWang3211/ViT-5/blob/main/vit5_base_patch16_224.pth) |
| ViT-5-Base  | 384 | 87M  | 85.4% | [Download](https://huggingface.co/FengWang3211/ViT-5/blob/main/vit5_base_patch16_384.pth) |
| ViT-5-Large | 224 | 304M | 84.9% | [Download](https://huggingface.co/FengWang3211/ViT-5/blob/main/vit5_large_patch16_224.pth) |
| ViT-5-Large | 384 | 304M | 86.0% | Available soon |

---

# Installation

```bash
# Install PyTorch (CUDA 12.4)
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
pip install timm==0.4.12 numpy==1.26.4 wandb einops

# Install NVIDIA Apex (required for fused optimizers)
git clone https://github.com/NVIDIA/apex
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

# (Optional) Install Flash Attention for faster training
pip install flash-attn==2.6.3 --no-build-isolation
```

---

# Training & Fine-tuning

```bash
# ImageNet Pretraining (8 GPUs example)
# ViT-5-Small
torchrun --nproc_per_node 8 main.py \
  --model vit5_small --input-size 224 --data-path YOUR_IMAGENET_PATH --output_dir DIR_TO_SAVE_LOG_AND_CKPT \
  --batch 256 --accum_iter 1 --lr 4e-3 --weight-decay 0.05 --epochs 800 --opt fusedlamb --unscale-lr \
  --mixup .8 --cutmix 1.0 --color-jitter 0.3 --drop-path 0.05 --reprob 0.0 --smoothing 0.0 --ThreeAugment \
  --repeated-aug --bce-loss --warmup-epochs 5 --eval-crop-ratio 1.0 --dist-eval --disable_wandb

# ViT-5-Base
torchrun --nproc_per_node 8 main.py \
  --model vit5_base --input-size 192 --data-path YOUR_IMAGENET_PATH --output_dir DIR_TO_SAVE_LOG_AND_CKPT \
  --batch 256 --accum_iter 1 --lr 3e-3 --weight-decay 0.05 --epochs 800 --opt fusedlamb --unscale-lr \
  --mixup .8 --cutmix 1.0 --color-jitter 0.3 --drop-path 0.2 --reprob 0.0 --smoothing 0.0 --ThreeAugment \
  --repeated-aug --bce-loss --warmup-epochs 5 --eval-crop-ratio 1.0 --dist-eval --disable_wandb

# ViT-5-Large
torchrun --nproc_per_node 8 main.py \
  --model vit5_large --input-size 192 --data-path YOUR_IMAGENET_PATH --output_dir DIR_TO_SAVE_LOG_AND_CKPT \
  --batch 256 --accum_iter 1 --lr 3e-3 --weight-decay 0.05 --epochs 400 --opt fusedlamb --unscale-lr \
  --mixup .8 --cutmix 1.0 --color-jitter 0.3 --drop-path 0.35 --reprob 0.0 --smoothing 0.0 --ThreeAugment \
  --repeated-aug --bce-loss --warmup-epochs 5 --eval-crop-ratio 1.0 --dist-eval --disable_wandb

# Fine-tuning from Pretrained Checkpoint
# ViT-5-Small
torchrun --nproc_per_node 8 main.py \
  --model vit5_small --finetune PATH_TO_YOUR_CKPT --data-path YOUR_DATASET_PATH --output_dir DIR_TO_SAVE_LOG_AND_CKPT \
  --batch 64 --lr 1e-5 --weight-decay 0.1 --epochs 20 --unscale-lr --aa rand-m9-mstd0.5-inc1 --drop-path 0.05 \
  --reprob 0.0 --smoothing 0.1 --no-repeated-aug --dist-eval --load_ema --eval-crop-ratio 1.0 --disable_wandb

# ViT-5-Base
torchrun --nproc_per_node 8 main.py \
  --model vit5_base --finetune PATH_TO_YOUR_CKPT --data-path YOUR_DATASET_PATH --output_dir DIR_TO_SAVE_LOG_AND_CKPT \
  --batch 64 --lr 1e-5 --weight-decay 0.1 --epochs 20 --unscale-lr --aa rand-m9-mstd0.5-inc1 --drop-path 0.25 \
  --reprob 0.0 --smoothing 0.1 --no-repeated-aug --dist-eval --load_ema --eval-crop-ratio 1.0 --disable_wandb

# ViT-5-Large
torchrun --nproc_per_node 8 main.py \
  --model vit5_large --finetune PATH_TO_YOUR_CKPT --data-path YOUR_DATASET_PATH --output_dir DIR_TO_SAVE_LOG_AND_CKPT \
  --batch 64 --lr 1e-5 --weight-decay 0.1 --epochs 20 --unscale-lr --aa rand-m9-mstd0.5-inc1 --drop-path 0.5 \
  --reprob 0.0 --smoothing 0.1 --no-repeated-aug --dist-eval --load_ema --eval-crop-ratio 1.0 --disable_wandb
```

---


# Citation

If you use ViT-5, please cite:

```
@article{wang2026vit5,
  title={ViT-5: Vision Transformers for The Mid-2020s},
  author={Wang, Feng and Ren, Sucheng and Zhang, Tiezheng and Neskovic, Predrag and Bhattad, Anand and Xie, Cihang and Yuille, Alan},
  journal={arXiv preprint arXiv:2602.08071},
  year={2026}
}
```

---

# Acknowledgement

This work builds upon the strong foundation of Vision Transformers and recent advances in scalable Transformer architectures. The code is basically built upon DeiT.

