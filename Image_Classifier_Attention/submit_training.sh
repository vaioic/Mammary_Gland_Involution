#!/bin/bash
#SBATCH --job-name=wsi_training
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=47          # 190 cores / 4 GPUs ≈ 47 cores per GPU
#SBATCH --gres=gpu:4                # 4 GPUs
#SBATCH --mem=0                     # Request all memory on node
#SBATCH --time=48:00:00             # Max runtime
#SBATCH --partition=gpu             # Adjust to your partition name

# Load required modules (adjust based on your HPC setup)
module purge
module load cuda/11.8
module load python/3.9
module load gcc/9.3.0

# Activate virtual environment
source /path/to/your/venv/bin/activate

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# Create log directory
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per GPU: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Training parameters
METADATA_CSV="/path/to/processed_patches/processed_metadata.csv"
CHECKPOINT_DIR="checkpoints/experiment_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=4                        # Per GPU
EPOCHS=50
LEARNING_RATE=1e-4
MAX_PATCHES=500
NUM_WORKERS=8                       # Per GPU

# Model parameters
BACKBONE="resnet50"                 # Options: resnet50, resnet34, vit_b_16
MIL_TYPE="gated"                    # Options: simple, gated
NUM_CLASSES=8

# Run training with DDP
echo "Starting training with DDP on 4 GPUs..."
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_ddp.py \
    --metadata_csv $METADATA_CSV \
    --checkpoint_dir $CHECKPOINT_DIR \
    --backbone $BACKBONE \
    --mil_type $MIL_TYPE \
    --num_classes $NUM_CLASSES \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_patches $MAX_PATCHES \
    --num_workers $NUM_WORKERS \
    --use_class_weights \
    --use_wandb \
    --project_name "wsi-classification" \
    --experiment_name "exp_$(date +%Y%m%d_%H%M%S)"

echo "Training completed!"
echo "Checkpoint directory: $CHECKPOINT_DIR"
