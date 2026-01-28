"""
Example End-to-End Workflow
Demonstrates complete pipeline from preprocessing to inference
"""

from pathlib import Path


def example_workflow():
    """
    Complete example workflow for WSI classification.
    Adjust paths and parameters for your specific use case.
    """
    
    print("=" * 60)
    print("WSI Classification Pipeline - Example Workflow")
    print("=" * 60)
    
    # ============================================================
    # STEP 1: Data Organization
    # ============================================================
    print("\n[Step 1] Organizing data...")
    
    # Your data structure should look like:
    # project/
    #   ├── svs_files/          # Your WSI files
    #   ├── tissue_masks/       # Binary masks of tissue regions
    #   └── metadata.csv        # Labels and metadata
    
    base_dir = Path("/path/to/your/project")
    svs_dir = base_dir / "svs_files"
    mask_dir = base_dir / "tissue_masks"
    metadata_csv = base_dir / "metadata.csv"
    
    # Example metadata.csv format:
    # slide_id,label,patient_id,cohort,age,sex
    # slide_001,0,patient_A,train,45,M
    # slide_002,3,patient_B,train,62,F
    
    print(f"  SVS directory: {svs_dir}")
    print(f"  Mask directory: {mask_dir}")
    print(f"  Metadata: {metadata_csv}")
    
    # ============================================================
    # STEP 2: Preprocessing
    # ============================================================
    print("\n[Step 2] Preprocessing WSI files...")
    print("  This extracts patches from gigapixel slides")
    
    preprocessing_cmd = f"""
python preprocessing.py \\
    --metadata_csv {metadata_csv} \\
    --svs_dir {svs_dir} \\
    --mask_dir {mask_dir} \\
    --output_dir processed_patches/ \\
    --patch_size 256 \\
    --magnification 20.0 \\
    --tissue_threshold 0.5
    """
    
    print("\nRun this command:")
    print(preprocessing_cmd)
    
    print("\n  For faster processing on HPC, you can parallelize by splitting the dataset:")
    parallel_example = """
# Split metadata into N chunks and process in parallel
# Example for 4 parallel jobs:
python preprocessing.py --metadata_csv metadata_chunk_1.csv ... &
python preprocessing.py --metadata_csv metadata_chunk_2.csv ... &
python preprocessing.py --metadata_csv metadata_chunk_3.csv ... &
python preprocessing.py --metadata_csv metadata_chunk_4.csv ... &
wait
    """
    print(parallel_example)
    
    # After preprocessing, you'll have:
    # processed_patches/
    #   ├── processed_metadata.csv
    #   ├── slide_001/
    #   │   └── slide_001_patches.h5
    #   ├── slide_002/
    #   │   └── slide_002_patches.h5
    #   └── ...
    
    # ============================================================
    # STEP 3: Training
    # ============================================================
    print("\n[Step 3] Training the model...")
    
    # Option A: Single GPU Training
    print("\n  Option A: Single GPU Training")
    single_gpu_cmd = """
python train.py \\
    --metadata_csv processed_patches/processed_metadata.csv \\
    --checkpoint_dir checkpoints/experiment_1 \\
    --backbone resnet50 \\
    --mil_type gated \\
    --num_classes 8 \\
    --batch_size 4 \\
    --epochs 50 \\
    --learning_rate 1e-4 \\
    --weight_decay 1e-4 \\
    --max_patches 500 \\
    --val_split 0.2 \\
    --patience 10 \\
    --use_class_weights \\
    --use_wandb \\
    --project_name wsi-classification \\
    --experiment_name exp1_resnet50
    """
    print(single_gpu_cmd)
    
    # Option B: Multi-GPU Training (4 GPUs)
    print("\n  Option B: Multi-GPU Training (4 GPUs)")
    multi_gpu_cmd = """
python -m torch.distributed.launch \\
    --nproc_per_node=4 \\
    --master_port=29500 \\
    train_ddp.py \\
    --metadata_csv processed_patches/processed_metadata.csv \\
    --checkpoint_dir checkpoints/experiment_1_ddp \\
    --backbone resnet50 \\
    --mil_type gated \\
    --num_classes 8 \\
    --batch_size 4 \\
    --epochs 50 \\
    --learning_rate 1e-4 \\
    --max_patches 500 \\
    --num_workers 8 \\
    --use_class_weights
    """
    print(multi_gpu_cmd)
    
    # Option C: HPC with SLURM
    print("\n  Option C: HPC with SLURM")
    slurm_cmd = """
# 1. Edit submit_training.sh with your paths
# 2. Submit job:
sbatch submit_training.sh

# 3. Monitor job:
squeue -u $USER
tail -f logs/train_JOBID.out
    """
    print(slurm_cmd)
    
    # ============================================================
    # STEP 4: Inference and Heatmap Generation
    # ============================================================
    print("\n[Step 4] Running inference and generating heatmaps...")
    
    inference_cmd = """
python inference.py \\
    --checkpoint checkpoints/experiment_1/checkpoint_best.pth \\
    --metadata_csv processed_patches/processed_metadata.csv \\
    --svs_dir svs_files/ \\
    --output_dir inference_results/ \\
    --backbone resnet50 \\
    --mil_type gated \\
    --num_classes 8 \\
    --class_names "Normal" "Cancer_Grade1" "Cancer_Grade2" "Cancer_Grade3" "Inflammation" "Necrosis" "Stroma" "Other" \\
    --max_slides 100
    """
    
    print(inference_cmd)
    
    print("\nOutputs:")
    print("  - inference_results/inference_results.csv    # Predictions")
    print("  - inference_results/*_heatmap.npy           # Attention maps")
    print("  - inference_results/*_visualization.png      # Visual overlays")
    
    # ============================================================
    # STEP 5: Evaluation
    # ============================================================
    print("\n[Step 5] Evaluating results...")
    
    evaluation_code = """
# Load results
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

results_df = pd.read_csv('inference_results/inference_results.csv')

# Calculate metrics
y_true = results_df['true_label'].values
y_pred = results_df['predicted_class'].values

# Classification report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')

# Per-class accuracy
for class_idx in range(8):
    class_mask = y_true == class_idx
    class_acc = (y_true[class_mask] == y_pred[class_mask]).mean()
    print(f"Class {class_idx} accuracy: {class_acc:.3f}")
    """
    
    print(evaluation_code)
    
    # ============================================================
    # Additional Tips
    # ============================================================
    print("\n" + "=" * 60)
    print("Additional Tips:")
    print("=" * 60)
    
    tips = """
1. **Data Quality**:
   - Ensure tissue masks are accurate
   - Check for artifacts in slides (pen marks, folds)
   - Balance classes if possible (or use class weights)

2. **Hyperparameter Tuning**:
   - Start with batch_size=4, max_patches=500
   - Try learning_rate in [1e-5, 1e-4, 1e-3]
   - Experiment with different backbones

3. **Monitoring**:
   - Use W&B for real-time monitoring
   - Check attention heatmaps during training
   - Monitor both accuracy and balanced accuracy

4. **Performance Optimization**:
   - Use NVMe/SSD for data storage
   - Increase num_workers if CPU is idle
   - Profile code to find bottlenecks

5. **Reproducibility**:
   - Set seeds (--seed 42)
   - Save all hyperparameters
   - Version control your code
   - Document data preprocessing steps

6. **Interpretability**:
   - Attention heatmaps show important regions
   - Compare heatmaps across classes
   - Validate with pathologist feedback
    """
    
    print(tips)
    
    # ============================================================
    # Quick Reference
    # ============================================================
    print("\n" + "=" * 60)
    print("Quick Reference - Model Architectures")
    print("=" * 60)
    
    architectures = """
| Backbone  | Parameters | Memory | Speed | Accuracy |
|-----------|-----------|--------|-------|----------|
| resnet34  | 21.8M     | Low    | Fast  | Good     |
| resnet50  | 25.6M     | Medium | Medium| Better   |
| vit_b_16  | 86M       | High   | Slow  | Best     |

Recommendation:
- Start with resnet50 + gated attention
- If memory limited: resnet34
- If accuracy critical: vit_b_16

MIL Types:
- simple: Basic attention, faster
- gated: Better performance, recommended
    """
    
    print(architectures)
    
    print("\n" + "=" * 60)
    print("Ready to start! Follow the steps above.")
    print("=" * 60)


if __name__ == "__main__":
    example_workflow()
