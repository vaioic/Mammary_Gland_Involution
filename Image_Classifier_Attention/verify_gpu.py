#!/usr/bin/env python3
"""
GPU Setup Verification Script
Checks if PyTorch can access GPU and provides diagnostic information
"""

import sys

def check_gpu_setup():
    """Comprehensive GPU setup verification."""
    
    print("=" * 70)
    print("GPU Setup Verification")
    print("=" * 70)
    
    # Check Python version
    print(f"\n📊 Python Information:")
    print(f"   Version: {sys.version}")
    print(f"   Executable: {sys.executable}")
    
    # Try importing torch
    try:
        import torch
    except ImportError:
        print("\n❌ ERROR: PyTorch not installed!")
        print("\nInstall PyTorch:")
        print("   For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("   For CPU: pip install torch")
        sys.exit(1)
    
    # PyTorch version
    print(f"\n📦 PyTorch Information:")
    print(f"   PyTorch version: {torch.__version__}")
    
    # Check torchvision
    try:
        import torchvision
        print(f"   Torchvision version: {torchvision.__version__}")
    except ImportError:
        print("   ⚠️  Torchvision not installed")
    
    # CUDA availability
    print(f"\n🎮 GPU Information:")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {cuda_available}")
    
    if cuda_available:
        # CUDA details
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
        
        # GPU count and details
        num_gpus = torch.cuda.device_count()
        print(f"\n   Number of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            print(f"\n   GPU {i}:")
            print(f"      Name: {torch.cuda.get_device_name(i)}")
            
            props = torch.cuda.get_device_properties(i)
            print(f"      Compute Capability: {props.major}.{props.minor}")
            print(f"      Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"      Multi-Processor Count: {props.multi_processor_count}")
            
            # Current memory usage
            if hasattr(torch.cuda, 'mem_get_info'):
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                print(f"      Memory Used: {used / 1024**3:.2f} GB / {total / 1024**3:.2f} GB")
        
        # Test GPU operations
        print(f"\n🧪 Testing GPU Operations:")
        try:
            print("   Creating tensors on GPU...")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            
            print("   Performing matrix multiplication...")
            z = torch.matmul(x, y)
            
            print("   Transferring result to CPU...")
            result = z.cpu()
            
            print("   ✅ GPU tensor operations successful!")
            
            # Benchmark
            import time
            print("\n   Quick benchmark:")
            
            # GPU
            start = time.time()
            for _ in range(100):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            print(f"      GPU: {gpu_time:.4f} seconds")
            
            # CPU
            x_cpu = x.cpu()
            y_cpu = y.cpu()
            start = time.time()
            for _ in range(100):
                z_cpu = torch.matmul(x_cpu, y_cpu)
            cpu_time = time.time() - start
            print(f"      CPU: {cpu_time:.4f} seconds")
            print(f"      Speedup: {cpu_time / gpu_time:.2f}x")
            
        except Exception as e:
            print(f"   ❌ GPU operation failed: {e}")
            cuda_available = False
    
    else:
        # GPU not available - provide diagnostics
        print("\n   ⚠️  GPU not detected!")
        print("\n   Possible reasons:")
        print("   1. PyTorch CPU-only version installed")
        print("   2. CUDA toolkit not installed")
        print("   3. NVIDIA drivers not installed")
        print("   4. No NVIDIA GPU in system")
        
        # Check if torch was compiled with CUDA
        print(f"\n   PyTorch built with CUDA: {torch.version.cuda is not None}")
        
        if torch.version.cuda is None:
            print("\n   ❌ You have CPU-only PyTorch!")
            print("\n   To install GPU-enabled PyTorch:")
            print("   1. Check CUDA version: nvcc --version")
            print("   2. Uninstall current PyTorch: pip uninstall torch torchvision")
            print("   3. Install GPU version:")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    # Final verdict
    print("\n" + "=" * 70)
    if cuda_available:
        print("✅ GPU SETUP SUCCESSFUL - Ready for training!")
        print("=" * 70)
        print("\nYou can now run:")
        print("   python train.py --metadata_csv data.csv")
        print("   python train_ddp.py  # For multi-GPU")
        return True
    else:
        print("⚠️  GPU NOT AVAILABLE - Will run on CPU only")
        print("=" * 70)
        print("\nYou can still run on CPU, but it will be much slower:")
        print("   python train.py --metadata_csv data.csv --batch_size 1")
        print("\nFor GPU support, see: GPU_SETUP_GUIDE.md")
        return False


def check_other_dependencies():
    """Check other required packages."""
    print("\n" + "=" * 70)
    print("Checking Other Dependencies")
    print("=" * 70)
    
    required = [
        'numpy',
        'pandas',
        'h5py',
        'cv2',
        'openslide',
        'albumentations',
        'tqdm',
        'sklearn'
    ]
    
    missing = []
    
    for package in required:
        try:
            if package == 'cv2':
                import cv2
                print(f"   ✅ opencv-python: {cv2.__version__}")
            elif package == 'sklearn':
                import sklearn
                print(f"   ✅ scikit-learn: {sklearn.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"   ✅ {package}: {version}")
        except ImportError:
            print(f"   ❌ {package}: Not installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def main():
    """Run all checks."""
    gpu_ok = check_gpu_setup()
    deps_ok = check_other_dependencies()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if gpu_ok and deps_ok:
        print("✅ System ready for GPU-accelerated training!")
        print("\nRecommended next steps:")
        print("1. Preprocess your WSI files: python preprocessing.py")
        print("2. Start training: python train.py --metadata_csv processed_metadata.csv")
        print("3. Use multiple GPUs: python train_ddp.py")
    elif deps_ok:
        print("⚠️  Dependencies OK, but GPU not available")
        print("   Training will work but will be slow on CPU")
        print("   See GPU_SETUP_GUIDE.md for GPU setup instructions")
    else:
        print("❌ Missing dependencies - install requirements.txt")
    
    print("=" * 70)
    
    sys.exit(0 if (gpu_ok or deps_ok) else 1)


if __name__ == "__main__":
    main()
