"""
Training script for WSI classification with single GPU support
Supports: class weighting, early stopping, learning rate scheduling

For multi-GPU training, use train_ddp.py instead
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import argparse
from typing import Dict, List

from model import create_model
from dataset import WSIDataset, get_transforms, collate_fn


class Trainer:
    """
    Trainer class for WSI classification with multi-GPU support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        device: torch.device,
        config: Dict,
        use_wandb: bool = False,
        rank: int = 0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.use_wandb = use_wandb and rank == 0
        self.rank = rank
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [Train]", disable=self.rank != 0)
        
        for _, (patches_list, labels, _, _) in enumerate(pbar):
            # Move data to device
            labels = labels.to(self.device)
            
            # Process each slide in the batch
            batch_logits = []
            for patches in patches_list:
                patches = patches.unsqueeze(0).to(self.device)  # Add batch dimension
                logits, _ = self.model(patches, return_attention=False)
                batch_logits.append(logits)
            
            batch_logits = torch.cat(batch_logits, dim=0)  # (batch_size, num_classes)
            
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
            
            # Update progress bar
            if self.rank == 0:
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
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
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Val]", disable=self.rank != 0)
        
        for patches_list, labels, _, _ in pbar:
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
            
            # Track metrics
            running_loss += loss.item()
            preds = torch.argmax(batch_logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'val_loss': avg_loss,
            'val_acc': accuracy,
            'val_balanced_acc': balanced_acc,
            'val_f1': f1,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'checkpoint_latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'checkpoint_best.pth')
            print(f"💾 Saved best model with val_acc: {metrics['val_acc']:.4f}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.config['scheduler_type'] == 'reduce_on_plateau':
                self.scheduler.step(val_metrics['val_loss'])
            else:
                self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'lr': self.optimizer.param_groups[0]['lr']}
            
            # Log to wandb
            if self.use_wandb:
                wandb.log(metrics, step=epoch)
            
            # Print metrics
            if self.rank == 0:
                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_acc']:.4f}")
                print(f"  Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.4f}")
                print(f"  Val Balanced Acc: {val_metrics['val_balanced_acc']:.4f} | Val F1: {val_metrics['val_f1']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['val_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['val_acc']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(metrics, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\nTraining completed. Best val_acc: {self.best_val_acc:.4f}")


def get_class_weights(metadata_df: pd.DataFrame, train_ids: List[str]) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets."""
    train_df = metadata_df[metadata_df['Image'].isin(train_ids)]
    class_counts = train_df['label'].value_counts().sort_index().values
    
    # Inverse frequency weighting
    total = len(train_df)
    weights = total / (len(class_counts) * class_counts)
    
    return torch.FloatTensor(weights)


def train_single_gpu(config: Dict):
    """Train on single GPU."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load metadata
    metadata_df = pd.read_csv(config['metadata_csv'])
    
    # Create train/val split (stratified)
    slide_ids = metadata_df['Image'].unique()
    labels = metadata_df.groupby('Image')['label'].first().values
    
    train_ids, val_ids = train_test_split(
        slide_ids,
        test_size=config['val_split'],
        stratify=labels,
        random_state=config['seed']
    )
    
    # Create datasets
    train_df = metadata_df[metadata_df['Image'].isin(train_ids)].reset_index(drop=True)
    val_df = metadata_df[metadata_df['Image'].isin(val_ids)].reset_index(drop=True)
    
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = create_model(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained'],
        mil_type=config['mil_type']
    ).to(device)
    
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
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project=config['project_name'],
            config=config,
            name=config['experiment_name']
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        use_wandb=config['use_wandb']
    )
    
    # Train
    trainer.train()
    
    if config['use_wandb']:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train WSI Classifier')
    
    # Data parameters
    parser.add_argument('--metadata_csv', type=str, required=True, help='Path to metadata CSV')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet34', 'vit_b_16'])
    parser.add_argument('--mil_type', type=str, default='gated', choices=['simple', 'gated'])
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
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'reduce_on_plateau', 'step'])
    
    # Data loading
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default='wsi-classification')
    parser.add_argument('--experiment_name', type=str, default='experiment')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Convert to config dict
    config = vars(args)
    
    # Train
    train_single_gpu(config)


if __name__ == "__main__":
    main()
