"""
Distributed Data Parallel (DDP) Training Script for Multi-GPU
Optimized for HPC with 4 GPUs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import argparse

from model import create_model
from dataset import WSIDataset, get_transforms, collate_fn
from train import Trainer, get_class_weights


def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    return rank, world_size, device


def cleanup_ddp():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def create_distributed_dataloaders(
    metadata_csv: str,
    train_ids: list,
    val_ids: list,
    config: dict,
    rank: int,
    world_size: int
):
    """Create distributed dataloaders with proper samplers."""
    metadata_df = pd.read_csv(metadata_csv)
    
    # Create datasets
    train_df = metadata_df[metadata_df['slide_id'].isin(train_ids)].reset_index(drop=True)
    val_df = metadata_df[metadata_df['slide_id'].isin(val_ids)].reset_index(drop=True)
    
    train_dataset = WSIDataset(
        train_df,
        transform=get_transforms(augment=True),
        max_patches=config['max_patches'],
        sampling_strategy='random'
    )
    
    val_dataset = WSIDataset(
        val_df,
        transform=get_transforms(augment=False),
        max_patches=config['max_patches'],
        sampling_strategy='random'
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    return train_loader, val_loader, train_sampler


class DDPTrainer(Trainer):
    """Extended Trainer for DDP with proper gradient synchronization."""
    
    def __init__(self, *args, train_sampler=None, world_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sampler = train_sampler
        self.world_size = world_size
    
    def train_epoch(self):
        """Train for one epoch with DDP."""
        self.model.train()
        
        # Set epoch for sampler (ensures different shuffle each epoch)
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch} [Train]",
            disable=self.rank != 0
        )
        
        for _, (patches_list, labels, _, _) in enumerate(pbar):
            labels = labels.to(self.device)
            
            # Process each slide in the batch
            batch_logits = []
            for patches in patches_list:
                patches = patches.unsqueeze(0).to(self.device)
                logits, _ = self.model(patches, return_attention=False)
                batch_logits.append(logits)
            
            batch_logits = torch.cat(batch_logits, dim=0)
            
            # Compute loss
            loss = self.criterion(batch_logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            preds = torch.argmax(batch_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            if self.rank == 0:
                pbar.set_postfix({'loss': loss.item()})
        
        # Gather metrics from all processes
        all_preds = torch.tensor(all_preds, device=self.device)
        all_labels = torch.tensor(all_labels, device=self.device)
        
        # Synchronize metrics across GPUs
        all_preds_gathered = [torch.zeros_like(all_preds) for _ in range(self.world_size)]
        all_labels_gathered = [torch.zeros_like(all_labels) for _ in range(self.world_size)]
        
        dist.all_gather(all_preds_gathered, all_preds)
        dist.all_gather(all_labels_gathered, all_labels)
        
        if self.rank == 0:
            all_preds = torch.cat(all_preds_gathered).cpu().numpy()
            all_labels = torch.cat(all_labels_gathered).cpu().numpy()
            
            avg_loss = running_loss / len(self.train_loader)
            accuracy = accuracy_score(all_labels, all_preds)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            
            metrics = {
                'train_loss': avg_loss,
                'train_acc': accuracy,
                'train_balanced_acc': balanced_acc,
                'train_f1': f1
            }
        else:
            metrics = {}
        
        # Broadcast metrics to all processes
        metrics_list = [metrics]
        dist.broadcast_object_list(metrics_list, src=0)
        
        return metrics_list[0]
    
    @torch.no_grad()
    def validate(self):
        """Validate with DDP."""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.epoch} [Val]",
            disable=self.rank != 0
        )
        
        for patches_list, labels, _, _ in pbar:
            labels = labels.to(self.device)
            
            batch_logits = []
            for patches in patches_list:
                patches = patches.unsqueeze(0).to(self.device)
                logits, _ = self.model(patches, return_attention=False)
                batch_logits.append(logits)
            
            batch_logits = torch.cat(batch_logits, dim=0)
            loss = self.criterion(batch_logits, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(batch_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # Gather metrics from all processes
        all_preds = torch.tensor(all_preds, device=self.device)
        all_labels = torch.tensor(all_labels, device=self.device)
        
        all_preds_gathered = [torch.zeros_like(all_preds) for _ in range(self.world_size)]
        all_labels_gathered = [torch.zeros_like(all_labels) for _ in range(self.world_size)]
        
        dist.all_gather(all_preds_gathered, all_preds)
        dist.all_gather(all_labels_gathered, all_labels)
        
        if self.rank == 0:
            all_preds = torch.cat(all_preds_gathered).cpu().numpy()
            all_labels = torch.cat(all_labels_gathered).cpu().numpy()
            
            avg_loss = running_loss / len(self.val_loader)
            accuracy = accuracy_score(all_labels, all_preds)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            
            metrics = {
                'val_loss': avg_loss,
                'val_acc': accuracy,
                'val_balanced_acc': balanced_acc,
                'val_f1': f1
            }
        else:
            metrics = {}
        
        # Broadcast metrics to all processes
        metrics_list = [metrics]
        dist.broadcast_object_list(metrics_list, src=0)
        
        return metrics_list[0]
    
    def save_checkpoint(self, metrics, is_best=False):
        """Only save on rank 0."""
        if self.rank != 0:
            return
        
        super().save_checkpoint(metrics, is_best)


def train_ddp(rank: int, world_size: int, config: dict):
    """Main DDP training function."""
    
    # Initialize DDP
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"Training on {world_size} GPUs")
        print(f"Configuration: {config}")
    
    # Set seeds for reproducibility
    torch.manual_seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    
    # Load metadata and create splits
    metadata_df = pd.read_csv(config['metadata_csv'])
    slide_ids = metadata_df['slide_id'].unique()
    labels = metadata_df.groupby('slide_id')['label'].first().values
    
    train_ids, val_ids = train_test_split(
        slide_ids,
        test_size=config['val_split'],
        stratify=labels,
        random_state=config['seed']
    )
    
    # Create distributed dataloaders
    train_loader, val_loader, train_sampler = create_distributed_dataloaders(
        config['metadata_csv'],
        train_ids,
        val_ids,
        config,
        rank,
        world_size
    )
    
    # Create model
    model = create_model(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained'],
        mil_type=config['mil_type']
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function with class weights
    if config['use_class_weights']:
        class_weights = get_class_weights(metadata_df, train_ids).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    if config['scheduler_type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs']
        )
    elif config['scheduler_type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Initialize wandb on rank 0
    if config['use_wandb'] and rank == 0:
        wandb.init(
            project=config['project_name'],
            config=config,
            name=config['experiment_name']
        )
    
    # Create trainer
    trainer = DDPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        use_wandb=config['use_wandb'],
        rank=rank,
        train_sampler=train_sampler,
        world_size=world_size
    )
    
    # Train
    trainer.train()
    
    if config['use_wandb'] and rank == 0:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='DDP Training for WSI Classifier')
    
    # Data parameters
    parser.add_argument('--metadata_csv', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--mil_type', type=str, default='gated')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--pretrained', action='store_true', default=True)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_patches', type=int, default=500)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    
    # Data loading
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default='wsi-classification')
    parser.add_argument('--experiment_name', type=str, default='experiment')
    
    # DDP
    parser.add_argument('--local_rank', type=int, default=-1)
    
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, device = setup_ddp()
    
    # Convert to config dict
    config = vars(args)
    
    try:
        # Train
        train_ddp(rank, world_size, config)
    finally:
        # Cleanup
        cleanup_ddp()


if __name__ == "__main__":
    main()
