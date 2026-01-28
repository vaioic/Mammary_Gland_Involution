# Deep Learning Classifier for Gigapixel Histology Images

## Overview

For gigapixel whole slide images (WSI), you need a **Multiple Instance Learning (MIL)** approach since you can't load the entire image into memory. The best architecture combines:
- **Patch-level feature extraction** (ResNet, ViT, or similar)
- **Attention-based aggregation** for interpretability
- **Slide-level classification**

## Architecture Recommendation

I recommend an **Attention-based MIL** approach:
1. Extract patches from WSI using tissue masks
2. Encode patches with a feature extractor (pre-trained ResNet50/ViT)
3. Use attention mechanism to aggregate patch features
4. The attention weights become your heatmap!

## Pipeline Steps

### 1. Data Preprocessing
### 2. Feature Extraction
### 3. Attention-based MIL Model
### 4. Training with Multi-GPU
### 5. Heatmap Generation

---

## Implementation

### Step 1: Environment Setup

```bash
# Install required packages
pip install openslide-python torch torchvision pandas numpy scikit-learn pillow h5py albumentations tqdm
```

### Step 2: WSI Preprocessing & Patch Extraction

This handles the gigabit SVS files by:
- Reading at appropriate magnification
- Using tissue masks to extract only relevant patches
- Saving patch coordinates for heatmap generation

**See `preprocessing.py` for complete implementation**

---

## Why Attention-based MIL?

Traditional approaches fail with gigapixel images because:
1. **Cannot fit in memory** - A 100,000 x 100,000 pixel image at 3 channels = 30GB
2. **Spatial relationships** - Patches don't exist in isolation
3. **Variable sizes** - Different slides have different numbers of patches

**Multiple Instance Learning (MIL)** solves this:
- Treats each slide as a "bag" of patches (instances)
- Only slide-level labels needed (not patch-level)
- Attention mechanism learns which patches matter

## Model Components

### 1. Feature Extractor
- **Input**: 256×256×3 patches
- **Architecture**: Pre-trained ResNet50, ResNet34, or ViT
- **Output**: Fixed-length feature vectors (e.g., 2048-dim for ResNet50)
- **Purpose**: Convert raw pixels to meaningful representations

### 2. Attention Mechanism
Two types available:

#### Simple Attention MIL
```python
attention_score = tanh(W * features)
attention_weight = softmax(attention_score)
slide_features = sum(attention_weight * features)
```

#### Gated Attention MIL (Recommended)
```python
V = tanh(W_V * features)
U = sigmoid(W_U * features)
attention_score = V ⊙ U  # element-wise multiplication (gating)
attention_weight = softmax(attention_score)
slide_features = sum(attention_weight * features)
```

The gating mechanism allows the model to filter out irrelevant features.

### 3. Classifier
- **Input**: Aggregated slide-level features
- **Architecture**: 2-layer MLP with ReLU and dropout
- **Output**: Class logits (8 classes in your case)

## Key Implementation Details

### Memory Management
- **Problem**: Cannot load all patches from a slide simultaneously
- **Solution**: Process patches in mini-batches, but aggregate at slide level
- **Trade-off**: `max_patches` parameter limits patches per slide
  - 500 patches = ~1-2 GB GPU memory per slide
  - During inference, can use all patches

### Class Imbalance
Histopathology datasets are often imbalanced. Handle with:
1. **Weighted Loss**: `weight = total_samples / (n_classes × class_count)`
2. **Balanced Accuracy**: Metric that accounts for imbalance
3. **Stratified Splitting**: Maintain class distribution in train/val/test

### Multi-GPU Training
Distributed Data Parallel (DDP) strategies:
- Each GPU processes different slides
- Gradients synchronized across GPUs
- Effective batch size = `batch_size × num_gpus`
- With 4 GPUs and batch_size=4: effective batch size = 16

### Learning Rate Scheduling
Recommended schedules:
1. **Cosine Annealing**: Smooth decay from max to min LR
2. **ReduceLROnPlateau**: Reduce LR when validation loss plateaus
3. **Warmup + Cosine**: Linear warmup for first few epochs

## Attention Heatmap Generation

The attention weights directly provide interpretability!

**Process**:
1. During inference, get attention weight for each patch
2. Map attention weights back to spatial coordinates
3. Create 2D heatmap by placing weights at patch locations
4. Smooth and normalize
5. Overlay on slide thumbnail with colormap

**Interpretation**:
- High attention (red) = important for classification
- Low attention (blue) = ignored by model
- Can validate findings with pathologist expertise

## Performance Optimization

### For HPC with 190 cores and 4 GPUs:

**Preprocessing** (CPU-intensive):
- Use all 190 cores for parallel patch extraction
- `num_workers = 180` (leave some for system)
- This stage is embarrassingly parallel

**Training** (GPU-intensive):
- 4 GPUs with DDP
- `num_workers = 8` per GPU (32 total for data loading)
- Remaining ~158 cores available for preprocessing next batch

**Optimal Workflow**:
```bash
# Terminal 1: Preprocessing (uses all 190 cores)
python preprocessing.py --num_workers 180

# Terminal 2: Training (uses 4 GPUs + 32 cores)
python train_ddp.py --num_workers 8
```

Or use SLURM to schedule both as separate jobs.

### Data Loading Bottlenecks
Common issues:
1. **Slow storage**: Use SSD/NVMe, not network drives
2. **CPU bottleneck**: Increase `num_workers`
3. **Memory bottleneck**: Reduce `max_patches` or `batch_size`

Check GPU utilization with `nvidia-smi`:
- Should be >90% during training
- If lower, data loading is bottleneck

## Validation Strategy

### Train/Val/Test Split
Recommended split: 70% / 15% / 15%

**CRITICAL**: Split at patient level, not slide level!
- Avoid data leakage from same patient in train/test
- Use stratified split to maintain class balance

### Cross-Validation
For smaller datasets (<500 slides):
```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
Train 5 models, report mean ± std of metrics

### Metrics to Track
- **Accuracy**: Overall correct predictions
- **Balanced Accuracy**: Accounts for class imbalance
- **F1 Score (Macro)**: Harmonic mean of precision/recall
- **Per-class Accuracy**: Identify difficult classes
- **Confusion Matrix**: See which classes are confused

## Common Issues and Solutions

### 1. Model Not Learning
**Symptoms**: Loss not decreasing, accuracy ~random
**Solutions**:
- Check data quality (labels correct?)
- Reduce learning rate (try 1e-5)
- Check if preprocessing worked (visualize patches)
- Try simpler model first (resnet34)

### 2. Overfitting
**Symptoms**: Train acc >> val acc
**Solutions**:
- Add more dropout (default 0.25, try 0.5)
- Data augmentation (already included)
- Reduce model complexity
- Early stopping (patience=10)
- Get more data

### 3. Attention Heatmaps Uninformative
**Symptoms**: All patches have similar attention
**Solutions**:
- Train longer (attention takes time to learn)
- Use gated attention instead of simple
- Increase hidden_dim (try 512)
- Check if model is learning (improving accuracy?)
- Validate on known cases (e.g., obvious tumor vs normal)

### 4. Training Very Slow
**Checks**:
```bash
# GPU utilization
nvidia-smi -l 1

# CPU usage
htop

# Data loading speed
# Add timing to dataloader
```

**Solutions**:
- Increase `num_workers`
- Use faster storage (SSD/NVMe)
- Profile code: `torch.profiler`
- Check `pin_memory=True`

## File Organization

```
project/
├── data/
│   ├── svs_files/
│   ├── tissue_masks/
│   └── metadata.csv
├── processed_patches/
│   ├── processed_metadata.csv
│   ├── slide_001/
│   │   └── slide_001_patches.h5
│   └── ...
├── checkpoints/
│   ├── experiment_1/
│   │   ├── checkpoint_best.pth
│   │   └── checkpoint_latest.pth
│   └── ...
├── inference_results/
│   ├── inference_results.csv
│   ├── slide_001_heatmap.npy
│   ├── slide_001_visualization.png
│   └── ...
├── preprocessing.py
├── model.py
├── dataset.py
├── train.py
├── train_ddp.py
├── inference.py
└── submit_training.sh
```

## Next Steps

1. **Preprocess your data**: Extract patches from SVS files
2. **Exploratory analysis**: Check class distribution, patch quality
3. **Start small**: Train on subset to validate pipeline
4. **Hyperparameter search**: Try different LR, architectures
5. **Full training**: Use all data with best config
6. **Validation**: Generate heatmaps, validate with experts
7. **Deploy**: Export model for clinical use

## References

**Key Papers**:
1. Ilse et al. "Attention-based Deep Multiple Instance Learning" (ICML 2018)
2. Lu et al. "Data-efficient and weakly supervised computational pathology" (Nature 2021)
3. Campanella et al. "Clinical-grade computational pathology" (Nature Medicine 2019)

**Resources**:
- OpenSlide: https://openslide.org/
- PyTorch DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- Histopathology datasets: TCGA, CPTAC, Camelyon

---

Good luck with your project! The complete implementation is in the accompanying Python files.
