# SegDINO: Few-Shot Segmentation with Enhanced DINOv3

> A few-shot segmentation framework based on DINOv3, integrating three enhancement modules (GLCM, Hypergraph GCN, Internal Adapter) for medical imaging (thyroid nodules), industrial defect detection (MVTec AD / ViSA), and remote sensing segmentation.

## 📌 Overview

This framework uses a frozen **DINOv3 ViT-S/16** as the backbone network, paired with a **DPT (Dense Prediction Transformer)** decoder head for dense prediction, enhanced by three plug-and-play lightweight modules:

| Module | Description | Parameters Ratio |
|---|---|---|
| **Internal Adapter** | Inject bottleneck adapters at specified Transformer layers to adapt frozen backbone to downstream data with minimal cost | ~0.5% |
| **GLCM** | Global-Local Calibration Module: Uses CLS token as normal baseline, identifies and enhances anomalous regions via patch-CLS similarity | ~1.5% |
| **Hypergraph GCN** | Hypergraph Convolutional Network: Builds high-order hypergraph relationships among patch features to capture long-range semantic dependencies | ~2.5% |

Architecture:

```
Input Image
    │
    ▼
DINOv3 ViT-S/16 (frozen)  ─── Internal Adapter (optional)
    │
    ├── Layer 6 features
    └── Layer 9 features
            │
            ▼
    ┌── GLCM (optional) ──┐
    │                      │
    ├── HyperGCN (opt.) ──┤
    │                      │
    └──────────────────────┘
            │
            ▼
      DPT Decoder Head
            │
            ▼
    Segmentation Map
```

## 📁 Project Structure

```
.
├── few_shot_tn3k.py              # Main training/evaluation script (supports TN3K / MVTec / ViSA / Remote Sensing)
├── few_shot_segdino_enhanced.py  # Enhanced multi-class segmentation training script
├── dpt.py                        # DPT decoder head (base version)
├── dpt_enhanced.py               # Enhanced DPT (integrated GLCM + HyperGCN + Adapter)
├── blocks.py                     # DPT building blocks
├── glcm_module.py                # GLCM global-local calibration module
├── hypergraph_module.py          # Hypergraph GCN module
├── internal_adapter.py           # Internal Adapter (injected within Transformer layers)
├── dataset.py                    # Generic folder dataset loader
├── dataset_mvtec.py              # MVTec AD dataset loader
├── calc_params.py                # Module parameter calculation script
├── requirements.txt              # Python dependencies
├── dinov3/                       # DINOv3 backbone source code
├── segdata/                      # Dataset directory (prepare yourself)
│   ├── tn3k/                     #   TN3K thyroid nodule dataset
│   ├── mvtec/                    #   MVTec AD industrial defect dataset
│   └── visa/                     #   ViSA industrial defect dataset
├── web_pth/                      # Pre-trained weights
│   └── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
└── runs/                         # Training outputs & logs
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place datasets in the `segdata/` directory with the following structure:

- **TN3K**: `segdata/tn3k/{train,test}/{images,mask}/`
- **MVTec AD**: `segdata/mvtec/{bottle,cable,...}/` ([Download](https://www.mvtec.com/company/research/datasets/mvtec-ad))
- **ViSA**: `segdata/visa/{candle,capsules,...}/`

### 3. Prepare Pre-trained Weights

Place DINOv3 ViT-S/16 pre-trained weights in the `web_pth/` directory:
```
web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

### 4. Training

**TN3K Thyroid Nodule Segmentation (5-shot):**
```bash
python few_shot_tn3k.py \
    --dataset_type tn3k \
    --data_dir ./segdata/tn3k \
    --k_shots 5 \
    --epochs 50 \
    --use_internal_adapter \
    --use_glcm \
    --use_hypergraph \
    --use_layers 6_9
```

**MVTec Industrial Defect Detection (all categories):**
```bash
python few_shot_tn3k.py \
    --dataset_type mvtec \
    --data_dir ./segdata/mvtec \
    --mvtec_category all \
    --k_shots 5 10 20
```

**ViSA Defect Detection (specific category):**
```bash
python few_shot_tn3k.py \
    --dataset_type visa \
    --data_dir ./segdata/visa \
    --visa_category candle \
    --k_shots 5
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_type` | `tn3k` | Dataset type: `tn3k` / `mvtec` / `visa` / `satellite` |
| `--k_shots` | `5 10 20` | Number of few-shot samples (multiple values for sequential experiments) |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--augment_factor` | `10` | Data augmentation factor |
| `--sampling_strategy` | `top` | Sampling strategy: `top` (max foreground) / `diverse` (uniform) |
| `--use_internal_adapter` | `false` | Enable Internal Adapter |
| `--use_glcm` | `false` | Enable GLCM module |
| `--use_hypergraph` | `false` | Enable Hypergraph GCN module |
| `--use_layers` | `6_9` | Feature layer selection: `6_9` (2 layers) / `all` (4 layers) |
| `--early_stopping_patience` | `10` | Early stopping patience |
| `--seed` | `42` | Random seed |

## 📊 Evaluation Metrics

- **Dice Coefficient**: Region overlap
- **HD95** (Hausdorff Distance 95%): Boundary distance

## 🔧 Technical Details

### Internal Adapter
- Uses bottleneck structure (384 → 64 → 384), injected into layers 3/6/9 of frozen Transformer via forward hooks
- Initialized to near-zero mapping to avoid disrupting pre-trained features during early training

### GLCM
- Uses CLS token as global "normal baseline", computes patch-CLS similarity for each patch
- Low-similarity regions are identified as anomalous and feature representations are enhanced

### Hypergraph GCN
- Constructs k-NN hypergraph based on patch feature similarity
- Aggregates high-order neighborhood information via hypergraph convolution, enhancing long-range semantic associations

## 📋 Dependencies

- Python ≥ 3.8
- PyTorch ≥ 2.0 (CUDA 12.1)
- torchvision
- NumPy, OpenCV, Pillow, scipy, scikit-learn
- tqdm, pandas, matplotlib

See [requirements.txt](requirements.txt) for the complete list.

## 📜 License

The DINOv3 backbone in this project follows its [original license](dinov3/LICENSE.md).
