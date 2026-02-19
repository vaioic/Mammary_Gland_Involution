"""
Dataset and DataLoader for WSI Classification
Handles loading preprocessed patches and efficient batching
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WSIDataset(Dataset):
    """
    Dataset for WSI classification.
    Loads patches from HDF5 files created during preprocessing.
    """
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        transform: Optional[A.Compose] = None,
        max_patches: Optional[int] = None,
        sampling_strategy: str = 'random'  # 'random' or 'all'
    ):
        """
        Args:
            metadata_df: DataFrame with columns ['slide_id', 'label', 'h5_path', ...]
            transform: Albumentations transform pipeline
            max_patches: Maximum number of patches per slide (for memory efficiency)
            sampling_strategy: How to sample patches if max_patches is set
        """
        self.metadata_df = metadata_df
        self.transform = transform
        self.max_patches = max_patches
        self.sampling_strategy = sampling_strategy
        
        # Verify all h5 files exist
        self.valid_indices = []
        for pos, (_, row) in enumerate(self.metadata_df.iterrows()):
            if Path(row['h5_path']).exists():
                self.valid_indices.append(pos)
        
        print(f"Found {len(self.valid_indices)} valid slides out of {len(self.metadata_df)}")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def load_patches(self, h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load patches and coordinates from HDF5 file.
        
        Returns:
            patches: (num_patches, H, W, C)
            coordinates: (num_patches, 2)
        """
        with h5py.File(h5_path, 'r') as f:
            patches = f['patches'][:]
            coordinates = f['coordinates'][:]
        
        return patches, coordinates
    
    def sample_patches(
        self,
        patches: np.ndarray,
        coordinates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample patches if max_patches is set.
        """
        num_patches = len(patches)
        
        if self.max_patches is None or num_patches <= self.max_patches:
            return patches, coordinates
        
        if self.sampling_strategy == 'random':
            indices = np.random.choice(num_patches, self.max_patches, replace=False)
        else:
            # Evenly spaced sampling
            indices = np.linspace(0, num_patches - 1, self.max_patches, dtype=int)
        
        return patches[indices], coordinates[indices]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, np.ndarray, str]:
        """
        Returns:
            patches: (num_patches, C, H, W) tensor
            label: int
            coordinates: (num_patches, 2) array
            slide_id: str
        """
        data_idx = self.valid_indices[idx]
        row = self.metadata_df.iloc[data_idx]
        
        # Load data
        patches, coordinates = self.load_patches(row['h5_path'])
        label = int(row['label'])
        slide_id = row['slide_id']
        
        # Sample patches if needed
        patches, coordinates = self.sample_patches(patches, coordinates)
        
        # Apply transforms
        if self.transform is not None:
            transformed_patches = []
            for patch in patches:
                transformed = self.transform(image=patch)
                transformed_patches.append(transformed['image'])
            patches = torch.stack(transformed_patches)
        else:
            # Convert to tensor
            patches = torch.from_numpy(patches).permute(0, 3, 1, 2).float() / 255.0
        
        return patches, label, coordinates, slide_id


def get_transforms(augment: bool = True) -> A.Compose:
    """
    Get augmentation pipeline.
    
    Args:
        augment: Whether to apply augmentations (True for train, False for val/test)
    
    Returns:
        Albumentations compose object
    """
    if augment:
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform


def collate_fn(batch):
    """
    Custom collate function to handle variable number of patches per slide.
    
    Args:
        batch: List of (patches, label, coordinates, slide_id)
    
    Returns:
        patches: List of tensors (each of shape [num_patches, C, H, W])
        labels: tensor of shape [batch_size]
        coordinates: List of arrays
        slide_ids: List of strings
    """
    patches_list = []
    labels = []
    coordinates_list = []
    slide_ids = []
    
    for patches, label, coords, slide_id in batch:
        patches_list.append(patches)
        labels.append(label)
        coordinates_list.append(coords)
        slide_ids.append(slide_id)
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return patches_list, labels, coordinates_list, slide_ids


def create_dataloaders(
    metadata_csv: str,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: Optional[List[str]] = None,
    batch_size: int = 4,
    num_workers: int = 8,
    max_patches: int = 500,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        metadata_csv: Path to processed metadata CSV
        train_ids: List of slide IDs for training
        val_ids: List of slide IDs for validation
        test_ids: List of slide IDs for testing (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_patches: Maximum patches per slide
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load metadata
    metadata_df = pd.read_csv(metadata_csv)
    
    # Split data
    train_df = metadata_df[metadata_df['slide_id'].isin(train_ids)].reset_index(drop=True)
    val_df = metadata_df[metadata_df['slide_id'].isin(val_ids)].reset_index(drop=True)
    
    # Create datasets
    train_dataset = WSIDataset(
        train_df,
        transform=get_transforms(augment=True),
        max_patches=max_patches,
        sampling_strategy='random'
    )
    
    val_dataset = WSIDataset(
        val_df,
        transform=get_transforms(augment=False),
        max_patches=max_patches,
        sampling_strategy='random'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = None
    if test_ids is not None:
        test_df = metadata_df[metadata_df['slide_id'].isin(test_ids)].reset_index(drop=True)
        test_dataset = WSIDataset(
            test_df,
            transform=get_transforms(augment=False),
            max_patches=None,  # Use all patches for testing
            sampling_strategy='all'
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one slide at a time for testing
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    metadata_csv = "processed_patches/processed_metadata.csv"
    
    # Example train/val split
    metadata_df = pd.read_csv(metadata_csv)
    all_slide_ids = metadata_df['slide_id'].unique().tolist()
    
    # Simple split (you should use stratified split in practice)
    train_size = int(0.7 * len(all_slide_ids))
    val_size = int(0.15 * len(all_slide_ids))
    
    train_ids = all_slide_ids[:train_size]
    val_ids = all_slide_ids[train_size:train_size + val_size]
    test_ids = all_slide_ids[train_size + val_size:]
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_csv=metadata_csv,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        batch_size=4,
        num_workers=8,
        max_patches=500
    )
    
    # Test loading a batch
    for patches_list, labels, coords_list, slide_ids in train_loader:
        print(f"Batch size: {len(patches_list)}")
        print(f"First slide patches shape: {patches_list[0].shape}")
        print(f"Labels: {labels}")
        print(f"Slide IDs: {slide_ids}")
        break
