"""
WSI Preprocessing: Extract patches from SVS files using tissue masks
Handles gigapixel images by tiling and filtering with masks
"""

import openslide
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
from tqdm import tqdm


class WSIPreprocessor:
    """
    Extracts patches from whole slide images using tissue masks.
    Saves patches and coordinates for downstream processing.
    """
    
    def __init__(
        self,
        patch_size: int = 256,
        target_magnification: float = 20.0,  # 20x magnification
        overlap: int = 0,
        tissue_threshold: float = 0.5,  # Minimum tissue coverage per patch
    ):
        self.patch_size = patch_size
        self.target_mag = target_magnification
        self.overlap = overlap
        self.tissue_threshold = tissue_threshold
        
    def get_magnification_level(self, slide: openslide.OpenSlide) -> int:
        """
        Find the pyramid level closest to target magnification.
        """
        objective_power = float(slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40
        ))
        
        # Find level with magnification closest to target
        best_level = 0
        best_diff = abs(objective_power - self.target_mag)
        
        for level in range(slide.level_count):
            downsample = slide.level_downsamples[level]
            level_mag = objective_power / downsample
            diff = abs(level_mag - self.target_mag)
            
            if diff < best_diff:
                best_diff = diff
                best_level = level
                
        return best_level
    
    def load_tissue_mask(self, mask_path: Path, slide_shape: Tuple[int,int]) -> np.ndarray:
        """
        Load and resize tissue mask to match slide dimensions at target level.
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask from {mask_path}")
        
        # Resize mask to match slide dimensions
        mask = cv2.resize(mask, (slide_shape[0], slide_shape[1]))
        return (mask > 0).astype(np.uint8)
    
    def generate_patch_coordinates(
        self,
        slide: openslide.OpenSlide,
        mask: np.ndarray,
        level: int
    ) -> List[Tuple[int, int]]:
        """
        Generate coordinates for patches that contain sufficient tissue.
        Returns coordinates at level 0 (highest resolution).
        """
        level_dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        
        # Calculate stride
        stride = self.patch_size - self.overlap
        
        coordinates = []
        
        for y in range(0, level_dims[1] - self.patch_size, stride):
            for x in range(0, level_dims[0] - self.patch_size, stride):
                # Check tissue coverage in mask
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                tissue_coverage = np.mean(mask_patch)
                
                if tissue_coverage >= self.tissue_threshold:
                    # Convert coordinates to level 0
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)
                    coordinates.append((x0, y0))
        
        return coordinates
    
    def extract_patch(
        self,
        slide: openslide.OpenSlide,
        coord: Tuple[int, int],
        level: int
    ) -> np.ndarray:
        """
        Extract a single patch from the slide.
        
        Args:
            coord: (x, y) coordinates at level 0
            level: pyramid level to read from
        """
        # OpenSlide.read_region takes level 0 coordinates,
        # but size parameter is at the specified level
        patch = slide.read_region(
            coord,
            level,
            (self.patch_size, self.patch_size)
        )
        
        # Convert RGBA to RGB
        patch = patch.convert('RGB')
        return np.array(patch)
    
    def process_single_slide(
        self,
        svs_path: Path,
        mask_path: Path,
        output_dir: Path,
        slide_id: str
    ) -> Dict:
        """
        Process a single WSI: extract patches and save with coordinates.
        """
        # Load slide
        slide = openslide.open_slide(str(svs_path))
        level = self.get_magnification_level(slide)
        level_dims = slide.level_dimensions[level]
        
        # Load tissue mask
        mask = self.load_tissue_mask(mask_path, level_dims)
        
        # Generate patch coordinates
        coordinates = self.generate_patch_coordinates(slide, mask, level)
        
        print(f"Extracting {len(coordinates)} patches from {slide_id}")
        
        # Create output directory
        slide_output_dir = Path(output_dir,str(slide_id))
        slide_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save patches and coordinates
        h5_path = Path(slide_output_dir,f"{slide_id}_patches.h5")
        
        with h5py.File(h5_path, 'w') as f:
            # Create dataset for patches
            patches_dataset = f.create_dataset(
                'patches',
                shape=(len(coordinates), self.patch_size, self.patch_size, 3),
                dtype=np.uint8,
                compression='gzip'
            )
            
            # Save coordinates
            f.create_dataset(
                'coordinates',
                data=np.array(coordinates),
                dtype=np.float32
            )
            
            # Extract and save patches
            for idx, coord in enumerate(tqdm(coordinates, desc=f"Processing {slide_id}")):
                patch = self.extract_patch(slide, coord, level)
                patches_dataset[idx] = patch
        
        slide.close()
        
        return {
            'Image': slide_id,
            'num_patches': len(coordinates),
            'h5_path': str(h5_path),
            'level': level,
            'magnification': self.target_mag
        }
    
    def process_dataset(
        self,
        metadata_df: pd.DataFrame,
        svs_dir: Path,
        mask_dir: Path,
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Process entire dataset of WSI files.
        
        Args:
            metadata_df: DataFrame with columns ['Image', 'label', ...]
            svs_dir: Directory containing .svs files
            mask_dir: Directory containing tissue masks
            output_dir: Directory to save processed patches
        
        Returns:
            Updated metadata DataFrame with processing information
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Process slides sequentially
        # Note: For parallel processing, run multiple instances of this script
        # with different subsets of metadata_df, as HDF5 writing is not thread-safe
        for _, row in metadata_df.iterrows():
            slide_id = row['Image']
            svs_path = Path(svs_dir,f"{slide_id}.svs")
            mask_path = Path(mask_dir,f"{slide_id}_mask.png") # Adjust extension as needed
            
            if not svs_path.exists():
                print(f"Warning: SVS file not found: {svs_path}")
                continue
            
            if not mask_path.exists():
                print(f"Warning: Mask file not found: {mask_path}")
                continue
            
            result = self.process_single_slide(
                svs_path,
                mask_path,
                output_dir,
                slide_id
            )
            results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original metadata
        output_df = metadata_df.merge(results_df, on='Image', how='left')
        
        # Save metadata
        output_df.to_csv(Path(output_dir, 'processed_metadata.csv'), index=False)
        
        return output_df


def main():
    """
    Example usage
    """
    # Paths - adjust to your setup
    metadata_path = Path("metadata.csv")  # CSV with slide_id and label columns
    svs_dir = Path("/path/to/svs/files")
    mask_dir = Path("/path/to/tissue/masks")
    output_dir = Path("processed_patches")
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    
    # Initialize preprocessor
    preprocessor = WSIPreprocessor(
        patch_size=256,
        target_magnification=20.0,
        overlap=0,
        tissue_threshold=0.5
    )
    
    # Process dataset
    processed_df = preprocessor.process_dataset(
        metadata_df,
        svs_dir,
        mask_dir,
        output_dir
    )
    
    print(f"Processed {len(processed_df)} slides")
    print(f"Total patches: {processed_df['num_patches'].sum()}")


if __name__ == "__main__":
    main()
