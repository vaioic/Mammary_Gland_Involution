# Test Pipeline Notebook Guide

## Overview

`test_pipeline.ipynb` is a comprehensive Jupyter notebook that tests every component of the WSI classification pipeline in a modular way. It's designed to:

1. ✅ Verify all functions work correctly
2. ✅ Demonstrate proper usage of each component
3. ✅ Provide visual feedback on results
4. ✅ Serve as a tutorial for understanding the pipeline
5. ✅ Help debug issues with real data

## What It Tests

### 1. Setup and Imports (Cell 1-2)
- Verifies all dependencies are installed
- Checks CUDA availability
- Creates temporary test directories
- Sets random seeds for reproducibility

### 2. Mock Data Generation (Cells 3-4)
- Generates synthetic histology patches that look like H&E stained tissue
- Creates HDF5 files mimicking preprocessed WSI data
- Produces metadata CSV with class labels
- **No real WSI files needed!**

### 3. Model Architecture Tests (Cells 5-7)
Tests each neural network component:
- **Feature Extractors**: ResNet34, ResNet50, ViT-B/16
- **Attention MIL**: Simple and Gated attention mechanisms
- **Complete WSI Classifier**: Full model integration

Verifies:
- Correct input/output shapes
- Parameter counts
- Forward pass execution

### 4. Dataset & DataLoader Tests (Cells 8-10)
- **Transforms**: Data augmentation and normalization
- **WSIDataset**: Loading patches from HDF5 files
- **DataLoader**: Batch collation with variable patch counts

Visual feedback:
- Shows original, augmented, and normalized patches
- Displays batch contents

### 5. Training Loop Tests (Cells 11-12)
- **Single Batch Overfitting**: Tests if model can learn
- **Gradient Flow**: Checks for vanishing/exploding gradients

Shows:
- Loss curves
- Training progress
- Gradient statistics

### 6. Inference & Heatmaps (Cells 13-15)
- **Model Inference**: Generate predictions
- **Attention Visualization**: Show which patches are important
- **Heatmap Generation**: Create spatial attention maps

Visualizations:
- Top patches by attention weight
- Attention distribution histograms
- Spatial heatmaps

### 7. End-to-End Integration (Cell 16)
- Complete training pipeline with train/val split
- 3-epoch training loop
- Plots training curves

Tests the entire workflow together!

### 8. Model Checkpointing (Cell 17)
- Saves model state
- Loads model state
- Verifies loaded model produces identical outputs

### 9. Cleanup (Cell 18)
- Removes temporary files
- Clears GPU memory

---

## How to Use

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install jupyter notebook
   pip install -r requirements.txt
   ```

2. **Launch notebook:**
   ```bash
   jupyter notebook test_pipeline.ipynb
   ```

3. **Run all cells:**
   - Click "Cell" → "Run All"
   - Or use Shift+Enter to run cell by cell
   - Watch the tests execute!

### Expected Runtime

| Hardware | Total Time |
|----------|-----------|
| GPU (CUDA) | ~5-10 minutes |
| CPU only | ~15-30 minutes |

### What to Look For

✅ **Success indicators:**
- All cells run without errors
- Loss decreases during training
- Attention weights sum to 1.0
- Heatmaps show spatial patterns
- Final message: "✅ All tests passed"

❌ **Failure indicators:**
- Import errors → Check dependencies
- Shape mismatches → Check model configuration
- CUDA errors → Check GPU availability
- Gradient NaN → Learning rate too high

---

## Troubleshooting

### "ModuleNotFoundError"
```bash
# Install missing package
pip install <package-name>
```

### "CUDA out of memory"
Modify these parameters in the notebook:
```python
# Reduce batch size
batch_size = 2  # instead of 4

# Reduce number of patches
num_patches = 20  # instead of 50

# Use smaller backbone
backbone = 'resnet34'  # instead of 'resnet50'
```

### "No module named 'model'"
Make sure all Python files are in the same directory as the notebook:
- model.py
- dataset.py
- preprocessing.py

### Tests fail with real data
The notebook uses **mock data** by default. To test with real data:
1. Run `preprocessing.py` on your WSI files first
2. Update paths in the notebook to your preprocessed data
3. Comment out the mock data generation cells

---

## Customization

### Test Different Backbones

```python
# In cell 5, modify:
backbones = ['resnet34', 'resnet50', 'vit_b_16']

# Or test just one:
backbones = ['vit_b_16']
```

### Test Different MIL Types

```python
# In cell 7, modify:
mil_type = 'simple'  # or 'gated'
```

### Change Number of Classes

```python
# In cell 4, modify:
num_classes = 4  # instead of 8
```

### Adjust Training Epochs

```python
# In cell 16, modify:
for epoch in range(10):  # instead of 3
```

---

## Understanding the Results

### Feature Extractor Output
```
Testing resnet50...
  Input shape: torch.Size([4, 3, 256, 256])
  Output shape: torch.Size([4, 2048])
  Feature dimension: 2048
  Parameters: 23,528,832
```

- **Input**: Batch of 4 patches, each 256×256×3
- **Output**: 4 feature vectors, each 2048-dimensional
- **This is correct!** Features are extracted successfully

### Attention Weights
```
Attention weights shape: torch.Size([2, 50])
Min: 0.015234
Max: 0.035678
Sum: 2.000000
```

- **Shape**: 2 slides, 50 patches each
- **Sum**: Should be ~2.0 (one per slide)
- **Range**: Usually 0.01-0.05 for 50 patches
- **This is correct!** Attention is distributed across patches

### Training Progress
```
Epoch 1/3:
  Train Loss: 2.1234, Train Acc: 0.1429
  Val Loss: 2.0987, Val Acc: 0.2000
```

- **Loss decreasing**: Model is learning! ✅
- **Accuracy improving**: Classifications getting better! ✅
- **Val loss < Train loss**: No overfitting (yet)

### Heatmap Interpretation
- **Red/Hot regions**: High attention → Important for classification
- **Blue/Cold regions**: Low attention → Less relevant
- **Clustered patterns**: Model finds coherent tissue regions
- **Uniform heatmap**: May need more training

---

## Advanced Usage

### Profile Performance

```python
# Add to any cell to measure time
import time

start = time.time()
# ... your code ...
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.2f} seconds")
```

### Debug Specific Components

Run individual sections independently:
```python
# Just test the dataset
from dataset import WSIDataset
# ... test code ...

# Just test the model
from model import create_model
# ... test code ...
```

### Export Results

```python
# Save attention heatmap
np.save('attention_heatmap.npy', heatmap)

# Save predictions
results_df.to_csv('predictions.csv', index=False)

# Save plots
fig.savefig('training_curves.png', dpi=300)
```

---

## Integration with Real Pipeline

After verifying with the test notebook:

1. **Preprocess real data:**
   ```bash
   python preprocessing.py --metadata_csv metadata.csv ...
   ```

2. **Train on real data:**
   ```bash
   python train.py --metadata_csv processed_metadata.csv ...
   ```

3. **Generate heatmaps:**
   ```bash
   python inference.py --checkpoint best_model.pth ...
   ```

The test notebook ensures all components work before using real WSI files!

---

## Benefits

✅ **No real data needed** - Tests with synthetic patches
✅ **Fast execution** - Runs in 5-10 minutes
✅ **Visual feedback** - See what each component does
✅ **Modular testing** - Test individual components
✅ **Educational** - Learn how each part works
✅ **Debugging** - Isolate issues before full training

---

## Summary

The test notebook provides:
- ✅ Confidence that all code works correctly
- ✅ Understanding of each pipeline component
- ✅ Visual validation of results
- ✅ Quick debugging of issues
- ✅ Template for experimentation

Run it whenever you:
- 🔧 Modify the code
- 📦 Install on a new system
- 🐛 Debug an issue
- 📚 Learn the pipeline
- 🧪 Test new features

**It's your pipeline health check!** 🏥🔬
