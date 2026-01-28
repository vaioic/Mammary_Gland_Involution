# GPU Setup Guide for WSI Classification Pipeline

## Overview

The default `requirements.txt` installs **CPU-only PyTorch**. For GPU acceleration, you need CUDA-enabled PyTorch.

**Key Point:** GPU support requires both:
1. ✅ **System-level**: CUDA Toolkit (NVIDIA drivers)
2. ✅ **Python-level**: CUDA-enabled PyTorch

---

## Quick Check: Do You Have GPU Support?

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("⚠️  GPU not available - running on CPU only")
```

If `CUDA available: False`, you need to install CUDA-enabled PyTorch.

---

## Installation Options

### Option 1: Local Machine (Recommended)

#### Step 1: Check Your CUDA Version

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit version
nvcc --version
```

**Common CUDA versions:**
- CUDA 11.8 (widely supported)
- CUDA 12.1 (latest)
- CUDA 11.7 (older systems)

#### Step 2: Install PyTorch with CUDA

Visit: https://pytorch.org/get-started/locally/

Or use these commands:

**For CUDA 11.8:**
```bash
pip uninstall torch torchvision  # Remove CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip uninstall torch torchvision  # Remove CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.7:**
```bash
pip uninstall torch torchvision  # Remove CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

#### Step 3: Install Other Requirements
```bash
pip install -r requirements.txt  # This won't reinstall PyTorch
```

#### Step 4: Verify GPU Access
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

### Option 2: HPC/Cluster (Your 4 GPU Setup)

Most HPC systems have CUDA pre-installed via modules.

#### Step 1: Load CUDA Module
```bash
# Check available modules
module avail cuda

# Load CUDA (adjust version to match available)
module load cuda/11.8
module load cudnn/8.6  # Optional but recommended

# Check loaded modules
module list
```

#### Step 2: Create Virtual Environment
```bash
# Create environment
python -m venv wsi_env
source wsi_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 3: Install PyTorch for Your CUDA Version
```bash
# Find your CUDA version
nvcc --version

# Install matching PyTorch (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Install Other Requirements
```bash
pip install -r requirements.txt
```

#### Step 5: Test GPU Access
```bash
# Request GPU node (adjust for your cluster)
salloc --gres=gpu:1 --time=00:10:00

# Test
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Exit
exit
```

---

### Option 3: Docker (Alternative)

Use NVIDIA's PyTorch container (CUDA already included):

```bash
# Pull PyTorch container with CUDA
docker pull nvcr.io/nvidia/pytorch:23.10-py3

# Run container with GPU access
docker run --gpus all -it -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.10-py3

# Inside container, install additional requirements
pip install openslide-python h5py albumentations wandb
```

---

## CUDA Version Compatibility

| PyTorch Version | CUDA 11.7 | CUDA 11.8 | CUDA 12.1 |
|-----------------|-----------|-----------|-----------|
| PyTorch 2.0.x | ✅ | ✅ | ✅ |
| PyTorch 2.1.x | ✅ | ✅ | ✅ |
| PyTorch 2.2.x | ❌ | ✅ | ✅ |

**Recommendation:** Use CUDA 11.8 for maximum compatibility.

---

## Verifying Your Installation

### Complete Verification Script

```python
import torch
import torchvision
import sys

print("=" * 60)
print("GPU Setup Verification")
print("=" * 60)

# Python version
print(f"\nPython version: {sys.version}")

# PyTorch version
print(f"\nPyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    # GPU details
    print(f"\nNumber of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
    
    # Test tensor operation
    print("\nTesting GPU tensor operation...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✅ GPU tensor operations working!")
    
    print("\n" + "=" * 60)
    print("✅ GPU SETUP SUCCESSFUL")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("⚠️  GPU NOT AVAILABLE")
    print("=" * 60)
    print("\nPossible reasons:")
    print("1. CUDA toolkit not installed")
    print("2. Wrong PyTorch version (CPU-only)")
    print("3. NVIDIA drivers not installed")
    print("4. GPU not detected by system")
    print("\nSee installation instructions above.")
```

Save as `verify_gpu.py` and run:
```bash
python verify_gpu.py
```

---

## System Requirements

### Minimum Requirements
- **GPU:** NVIDIA GPU with Compute Capability ≥ 3.5
- **VRAM:** 8 GB minimum (16+ GB recommended for WSI)
- **CUDA:** Version 11.7 or later
- **Driver:** NVIDIA driver ≥ 450.80.02

### Recommended for WSI Classification
- **GPU:** NVIDIA RTX 3090, A100, or similar
- **VRAM:** 24 GB or more
- **CUDA:** 11.8 or 12.1
- **System RAM:** 64+ GB (for large WSI files)

### Your HPC Setup (4 GPUs)
Based on your HPC description:
- **GPUs:** 4x (likely A100 or V100)
- **Cores:** 190 CPU cores
- **Perfect for:** Multi-GPU training with DDP

---

## Common Issues & Solutions

### Issue 1: `torch.cuda.is_available()` returns False

**Causes:**
- CPU-only PyTorch installed
- CUDA version mismatch
- No NVIDIA driver

**Solution:**
```bash
# Check current PyTorch
pip show torch | grep Version

# Check if CUDA build
python -c "import torch; print(torch.version.cuda)"

# Reinstall with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: CUDA out of memory

**Solutions:**
```python
# In train.py, reduce:
--batch_size 2          # Instead of 4
--max_patches 300       # Instead of 500

# Use smaller backbone
--backbone resnet34     # Instead of resnet50
```

### Issue 3: Slow training despite GPU

**Causes:**
- Data loading bottleneck
- CPU preprocessing slow

**Solutions:**
```bash
# Increase workers
--num_workers 8         # Or more

# Use faster storage (NVMe/SSD)
# Enable pin_memory in dataloaders (already enabled)
```

### Issue 4: Version conflicts

```bash
# Create fresh environment
python -m venv wsi_gpu_env
source wsi_gpu_env/bin/activate

# Install in order
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Issue 5: Multiple CUDA versions

**HPC systems often have multiple CUDA versions:**

```bash
# List available
module avail cuda

# Load specific version
module load cuda/11.8

# Verify
echo $CUDA_HOME
nvcc --version

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Performance Expectations

### Single GPU (e.g., RTX 3090)
- **Training speed:** ~100-150 slides/hour
- **Batch size:** 4-8
- **Max patches:** 500

### 4 GPUs with DDP (Your HPC)
- **Training speed:** ~400-600 slides/hour
- **Batch size:** 4 per GPU (16 effective)
- **Max patches:** 500
- **Use:** `train_ddp.py` script

### CPU Only (No GPU)
- **Training speed:** ~5-10 slides/hour
- **Batch size:** 1-2
- **Not recommended for production**

---

## HPC-Specific Setup

### Complete HPC Setup Script

```bash
#!/bin/bash
# save as: setup_gpu_hpc.sh

# Load modules
module purge
module load cuda/11.8
module load cudnn/8.6
module load python/3.9
module load gcc/9.3.0

# Create virtual environment
python -m venv wsi_gpu_env
source wsi_gpu_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Verify
python verify_gpu.py

echo "Setup complete! Run: source wsi_gpu_env/bin/activate"
```

Make executable and run:
```bash
chmod +x setup_gpu_hpc.sh
./setup_gpu_hpc.sh
```

### SLURM Job with GPU

The provided `submit_training.sh` already includes:
```bash
#SBATCH --gres=gpu:4                # Request 4 GPUs
```

This is correct! Just ensure you:
1. Load CUDA module in the script
2. Activate environment with GPU-enabled PyTorch

---

## Summary - What You Need

### For Local Development
1. ✅ NVIDIA GPU with ≥8 GB VRAM
2. ✅ CUDA Toolkit (11.8 recommended)
3. ✅ Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
4. ✅ Install other requirements: `pip install -r requirements.txt`
5. ✅ Verify: `python verify_gpu.py`

### For Your HPC (4 GPUs)
1. ✅ Load CUDA module: `module load cuda/11.8`
2. ✅ Create venv with GPU PyTorch
3. ✅ Use `train_ddp.py` for multi-GPU
4. ✅ Submit with `sbatch submit_training.sh`

### Quick Command Reference

```bash
# Check GPU
nvidia-smi

# Install GPU PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"

# Train with GPU
python train.py --metadata_csv data.csv --batch_size 4

# Train with 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 train_ddp.py
```

---

## Need Help?

**Check your setup:**
```bash
python verify_gpu.py
```

**If GPU not detected:**
1. Install NVIDIA drivers
2. Install CUDA toolkit
3. Install CUDA-enabled PyTorch (not CPU version)

**For HPC issues:**
1. Check available CUDA modules: `module avail cuda`
2. Ensure GPU allocation in SLURM: `#SBATCH --gres=gpu:4`
3. Verify GPU access in job: `nvidia-smi`

---

**Your HPC is ready for GPU training!** Just make sure to install CUDA-enabled PyTorch. 🚀
