# GPU Setup Quick Reference Card

## 🚀 Quick Start (Most Common)

```bash
# 1. Check CUDA version
nvcc --version

# 2. Install PyTorch with CUDA 11.8 (most compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install other requirements
pip install -r requirements.txt

# 4. Verify
python verify_gpu.py
```

---

## 💡 By Platform

### Local Machine with NVIDIA GPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python verify_gpu.py
```

### HPC/Cluster (Your 4 GPU Setup)
```bash
module load cuda/11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python verify_gpu.py
```

### Google Colab
```python
# GPU already configured, just install requirements
!pip install -r requirements.txt
```

### Docker
```bash
docker run --gpus all -it -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.10-py3
pip install openslide-python h5py albumentations wandb
```

---

## 🔍 Quick Checks

### Check if GPU available
```python
import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # Your GPU name
```

### Check CUDA version
```bash
nvidia-smi                  # Shows GPU status
nvcc --version             # Shows CUDA toolkit version
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA version
```

### Check GPU memory
```bash
nvidia-smi                 # Shows memory usage
```

---

## 📦 Installation URLs by CUDA Version

| CUDA Version | Installation Command |
|--------------|---------------------|
| 11.8 (recommended) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 (latest) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| 11.7 (older) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117` |

---

## ⚠️ Common Issues

### Issue: `torch.cuda.is_available()` returns False
**Fix:**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: CUDA out of memory
**Fix:** Reduce batch size and patches
```bash
python train.py --batch_size 2 --max_patches 300
```

### Issue: ModuleNotFoundError: No module named 'torch'
**Fix:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎯 Performance Expectations

| Setup | Speed | Best For |
|-------|-------|----------|
| CPU only | 5-10 slides/hr | Testing only |
| 1 GPU | 100-150 slides/hr | Development |
| 4 GPUs (Your HPC) | 400-600 slides/hr | Production |

---

## 📊 Recommended Settings

### For 1 GPU (e.g., RTX 3090)
```bash
python train.py \
    --batch_size 4 \
    --max_patches 500 \
    --num_workers 8
```

### For 4 GPUs (Your HPC)
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_ddp.py \
    --batch_size 4 \
    --max_patches 500 \
    --num_workers 8
```

---

## 📚 Documentation

- **Full Guide:** [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md)
- **Verification:** Run `python verify_gpu.py`
- **PyTorch Docs:** https://pytorch.org/get-started/locally/

---

## ✅ Verification Checklist

- [ ] CUDA toolkit installed (`nvcc --version`)
- [ ] NVIDIA driver installed (`nvidia-smi`)
- [ ] PyTorch with CUDA installed (not CPU version)
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] `verify_gpu.py` passes all checks
- [ ] GPU shows up in `nvidia-smi`

---

**Need help?** See [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md) for detailed instructions.
