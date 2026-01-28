"""
Inference script for WSI classification with attention heatmap generation
Generates class predictions and visualization of important regions
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import argparse
from tqdm import tqdm

from model import create_model
from dataset import get_transforms
import openslide


class WSIInference:
    """
    Inference class for WSI classification with interpretability.
    Generates predictions and attention heatmaps.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        patch_size: int = 256,
        class_names: List[str] = None
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.patch_size = patch_size
        self.class_names = class_names or [f"Class {i}" for i in range(8)]
        self.transform = get_transforms(augment=False)
    
    @torch.no_grad()
    def predict_slide(
        self,
        patches: np.ndarray
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Predict class for a slide and get attention weights.
        
        Args:
            patches: (num_patches, H, W, C) numpy array
        
        Returns:
            predicted_class: int
            probabilities: (num_classes,) array
            attention_weights: (num_patches,) array
        """
        # Transform patches
        transformed_patches = []
        for patch in patches:
            transformed = self.transform(image=patch)
            transformed_patches.append(transformed['image'])
        
        patches_tensor = torch.stack(transformed_patches).unsqueeze(0).to(self.device)
        
        # Forward pass
        logits, attention = self.model(patches_tensor, return_attention=True)
        
        # Get predictions
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        predicted_class = np.argmax(probabilities)
        attention_weights = attention.squeeze(0).cpu().numpy()
        
        return predicted_class, probabilities, attention_weights
    
    def create_heatmap(
        self,
        attention_weights: np.ndarray,
        coordinates: np.ndarray,
        slide_dimensions: Tuple[int, int],
        downsample: int = 32
    ) -> np.ndarray:
        """
        Create attention heatmap overlaid on slide thumbnail.
        
        Args:
            attention_weights: (num_patches,) attention weights
            coordinates: (num_patches, 2) patch coordinates at level 0
            slide_dimensions: (width, height) of slide at level 0
            downsample: Downsampling factor for heatmap resolution
        
        Returns:
            heatmap: (H, W) heatmap array
        """
        # Calculate heatmap dimensions
        heatmap_width = slide_dimensions[0] // downsample
        heatmap_height = slide_dimensions[1] // downsample
        
        # Initialize heatmap
        heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
        counts = np.zeros((heatmap_height, heatmap_width), dtype=np.int32)
        
        # Fill heatmap with attention weights
        patch_size_downsampled = self.patch_size // downsample
        
        for (x, y), weight in zip(coordinates, attention_weights):
            # Convert to downsampled coordinates
            x_down = x // downsample
            y_down = y // downsample
            
            # Get patch region in heatmap
            x_end = min(x_down + patch_size_downsampled, heatmap_width)
            y_end = min(y_down + patch_size_downsampled, heatmap_height)
            
            # Add attention weight to heatmap
            heatmap[y_down:y_end, x_down:x_end] += weight
            counts[y_down:y_end, x_down:x_end] += 1
        
        # Average where multiple patches overlap
        mask = counts > 0
        heatmap[mask] = heatmap[mask] / counts[mask]
        
        return heatmap
    
    def visualize_attention(
        self,
        slide_path: str,
        heatmap: np.ndarray,
        predicted_class: int,
        probabilities: np.ndarray,
        output_path: str,
        level: int = 4
    ):
        """
        Create visualization of attention heatmap on slide thumbnail.
        
        Args:
            slide_path: Path to SVS file
            heatmap: Attention heatmap
            predicted_class: Predicted class index
            probabilities: Class probabilities
            output_path: Where to save visualization
            level: Pyramid level for thumbnail
        """
        # Load slide thumbnail
        slide = openslide.open_slide(slide_path)
        thumbnail = slide.read_region(
            (0, 0),
            level,
            slide.level_dimensions[level]
        ).convert('RGB')
        thumbnail = np.array(thumbnail)
        
        # Resize heatmap to match thumbnail
        heatmap_resized = cv2.resize(
            heatmap,
            (thumbnail.shape[1], thumbnail.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize heatmap
        heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
        
        # Apply colormap
        heatmap_colored = plt.cm.jet(heatmap_normalized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Create overlay
        alpha = 0.4
        overlay = cv2.addWeighted(thumbnail, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 6))
        
        # Original thumbnail
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(thumbnail)
        ax1.set_title('Original Slide', fontsize=14)
        ax1.axis('off')
        
        # Attention heatmap
        ax2 = plt.subplot(1, 4, 2)
        im = ax2.imshow(heatmap_resized, cmap='jet')
        ax2.set_title('Attention Heatmap', fontsize=14)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046)
        
        # Overlay
        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(overlay)
        ax3.set_title(f'Overlay (Predicted: {self.class_names[predicted_class]})', fontsize=14)
        ax3.axis('off')
        
        # Class probabilities
        ax4 = plt.subplot(1, 4, 4)
        bars = ax4.barh(self.class_names, probabilities)
        
        # Color the predicted class differently
        bars[predicted_class].set_color('red')
        
        ax4.set_xlabel('Probability', fontsize=12)
        ax4.set_title('Class Probabilities', fontsize=14)
        ax4.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        slide.close()
    
    def process_slide(
        self,
        h5_path: str,
        svs_path: str,
        output_dir: Path,
        slide_id: str,
        true_label: int = None
    ) -> Dict:
        """
        Process a single slide: predict and visualize.
        
        Args:
            h5_path: Path to HDF5 file with patches
            svs_path: Path to original SVS file
            output_dir: Directory to save results
            slide_id: Slide identifier
            true_label: Ground truth label (optional)
        
        Returns:
            Dictionary with results
        """
        # Load patches and coordinates
        with h5py.File(h5_path, 'r') as f:
            patches = f['patches'][:]
            coordinates = f['coordinates'][:]
        
        # Get prediction and attention
        predicted_class, probabilities, attention_weights = self.predict_slide(patches)
        
        # Load slide for dimensions
        slide = openslide.open_slide(svs_path)
        slide_dimensions = slide.level_dimensions[0]
        slide.close()
        
        # Create heatmap
        heatmap = self.create_heatmap(
            attention_weights,
            coordinates,
            slide_dimensions,
            downsample=32
        )
        
        # Save heatmap as numpy array
        heatmap_path = output_dir / f"{slide_id}_heatmap.npy"
        np.save(heatmap_path, heatmap)
        
        # Create visualization
        viz_path = output_dir / f"{slide_id}_visualization.png"
        self.visualize_attention(
            svs_path,
            heatmap,
            predicted_class,
            probabilities,
            str(viz_path)
        )
        
        # Compile results
        results = {
            'slide_id': slide_id,
            'predicted_class': int(predicted_class),
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': float(probabilities[predicted_class]),
            'probabilities': probabilities.tolist()
        }
        
        if true_label is not None:
            results['true_label'] = int(true_label)
            results['true_label_name'] = self.class_names[true_label]
            results['correct'] = int(predicted_class) == int(true_label)
        
        return results


def load_model(checkpoint_path: str, config: Dict, device: torch.device):
    """Load trained model from checkpoint."""
    model = create_model(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=False,
        mil_type=config.get('mil_type', 'gated')
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def main():
    parser = argparse.ArgumentParser(description='WSI Inference with Attention Heatmaps')
    
    # Paths
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--metadata_csv', type=str, required=True, help='Path to metadata CSV')
    parser.add_argument('--svs_dir', type=str, required=True, help='Directory with SVS files')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='Output directory')
    
    # Model config
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--mil_type', type=str, default='gated')
    parser.add_argument('--num_classes', type=int, default=8)
    
    # Class names (optional)
    parser.add_argument('--class_names', type=str, nargs='+', help='Class names for visualization')
    
    # Subset of slides to process
    parser.add_argument('--slide_ids', type=str, nargs='+', help='Specific slide IDs to process')
    parser.add_argument('--max_slides', type=int, help='Maximum number of slides to process')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    config = {
        'backbone': args.backbone,
        'mil_type': args.mil_type,
        'num_classes': args.num_classes
    }
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config, device)
    
    # Initialize inference
    inference = WSIInference(
        model=model,
        device=device,
        patch_size=256,
        class_names=args.class_names
    )
    
    # Load metadata
    metadata_df = pd.read_csv(args.metadata_csv)
    
    # Filter slides if specified
    if args.slide_ids:
        metadata_df = metadata_df[metadata_df['slide_id'].isin(args.slide_ids)]
    
    if args.max_slides:
        metadata_df = metadata_df.head(args.max_slides)
    
    # Process slides
    results_list = []
    
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing slides"):
        slide_id = row['slide_id']
        h5_path = row['h5_path']
        svs_path = Path(args.svs_dir) / f"{slide_id}.svs"
        
        if not Path(h5_path).exists():
            print(f"Warning: HDF5 file not found: {h5_path}")
            continue
        
        if not svs_path.exists():
            print(f"Warning: SVS file not found: {svs_path}")
            continue
        
        try:
            results = inference.process_slide(
                h5_path=h5_path,
                svs_path=str(svs_path),
                output_dir=output_dir,
                slide_id=slide_id,
                true_label=row.get('label')
            )
            results_list.append(results)
            
        except Exception as e:
            print(f"Error processing {slide_id}: {e}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results_list)
    results_csv = output_dir / 'inference_results.csv'
    results_df.to_csv(results_csv, index=False)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Inference Summary")
    print(f"{'='*50}")
    print(f"Total slides processed: {len(results_df)}")
    
    if 'correct' in results_df.columns:
        accuracy = results_df['correct'].mean()
        print(f"Accuracy: {accuracy:.4f}")
        
        # Per-class accuracy
        print(f"\nPer-class accuracy:")
        for class_idx in range(args.num_classes):
            class_df = results_df[results_df['true_label'] == class_idx]
            if len(class_df) > 0:
                class_acc = class_df['correct'].mean()
                class_name = args.class_names[class_idx] if args.class_names else f"Class {class_idx}"
                print(f"  {class_name}: {class_acc:.4f} ({len(class_df)} slides)")
    
    print(f"\nResults saved to: {results_csv}")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
