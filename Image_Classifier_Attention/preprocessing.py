"""
WSI Preprocessing: Extract patches from SVS files using tissue masks.

Parallelisation strategy
------------------------
1. **Across slides** — ``ProcessPoolExecutor`` runs N slides simultaneously.
   Each slide writes to its own HDF5 file so there are no write conflicts.

2. **Within a slide** — ``ThreadPoolExecutor`` reads patches concurrently.
   OpenSlide is thread-safe for reads; we collect results in order then
   write to HDF5 serially (h5py is NOT thread-safe for writes).

3. **GPU** — Patch extraction is I/O / CPU bound so a GPU gives no benefit
   there. An optional Macenko colour-normalisation step (disabled by default)
   can run on GPU via ``torchstain`` if you need it.

Multi-object mask convention
-----------------------------
Pixel value 0 = background.
Values 1-n  = distinct objects; each is cropped with padding and processed
              independently, producing one HDF5 file per object per slide.
"""

import os
import openslide
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import cv2
from tqdm import tqdm
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ObjectRegion:
    """Bounding-box and binary mask for a single labelled object."""
    object_id: int
    x: int          # Left edge at level resolution (with padding)
    y: int          # Top edge at level resolution (with padding)
    w: int          # Width  at level resolution
    h: int          # Height at level resolution
    binary_mask: np.ndarray = field(repr=False)   # Cropped binary mask


# ---------------------------------------------------------------------------
# Worker-level helpers (must be picklable - defined at module scope)
# ---------------------------------------------------------------------------

def _read_patch_worker(
    svs_path: str,
    slide_x: int,
    slide_y: int,
    level: int,
    patch_size: int,
) -> Tuple[int, int, np.ndarray]:
    """
    Open the slide, read one patch, close.  Called from a thread pool.
    Returns (slide_x, slide_y, rgb_array).  OpenSlide handles are
    not shared across threads - each call opens its own handle.
    """
    slide = openslide.open_slide(svs_path)
    patch = slide.read_region((slide_x, slide_y), level, (patch_size, patch_size))
    slide.close()
    return slide_x, slide_y, np.array(patch.convert('RGB'))


def _process_slide_worker(args: Dict) -> List[Dict]:
    """
    Top-level worker function for ProcessPoolExecutor.
    Unpacks args and delegates to WSIPreprocessor.process_single_slide().
    Must be defined at module scope so it is picklable.
    """
    preprocessor = WSIPreprocessor(
        patch_size=args['patch_size'],
        target_magnification=args['target_mag'],
        overlap=args['overlap'],
        tissue_threshold=args['tissue_threshold'],
        object_padding=args['object_padding'],
        patch_read_workers=args['patch_read_workers'],
    )
    return preprocessor.process_single_slide(
        svs_path=Path(args['svs_path']),
        mask_path=Path(args['mask_path']),
        output_dir=Path(args['output_dir']),
        slide_id=args['slide_id'],
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WSIPreprocessor:
    """
    Extracts patches from whole slide images using (possibly multi-object)
    tissue masks with multi-process / multi-thread acceleration.

    Parameters
    ----------
    patch_size : int
        Side length of extracted square patches (pixels).
    target_magnification : float
        Desired magnification level (e.g. 20.0 for 20x).
    overlap : int
        Pixel overlap between adjacent patches.
    tissue_threshold : float
        Minimum fraction of a patch that must be tissue (0-1).
    object_padding : int
        Extra pixels added around each detected object bounding-box.
    patch_read_workers : int
        Threads used to read patches *within* a single slide.
        Defaults to min(32, os.cpu_count()).
        Set to 1 to disable intra-slide threading.
    """

    def __init__(
        self,
        patch_size: int = 256,
        target_magnification: float = 20.0,
        overlap: int = 0,
        tissue_threshold: float = 0.5,
        object_padding: int = 64,
        patch_read_workers: int = 0,       # 0 = auto
    ):
        self.patch_size = patch_size
        self.target_mag = target_magnification
        self.overlap = overlap
        self.tissue_threshold = tissue_threshold
        self.object_padding = object_padding
        self.patch_read_workers = patch_read_workers or min(32, os.cpu_count() or 4)

    # ------------------------------------------------------------------
    # Slide helpers
    # ------------------------------------------------------------------

    def get_magnification_level(self, slide: openslide.OpenSlide) -> int:
        objective_power = float(slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40
        ))
        best_level, best_diff = 0, abs(objective_power - self.target_mag)
        for level in range(slide.level_count):
            level_mag = objective_power / slide.level_downsamples[level]
            diff = abs(level_mag - self.target_mag)
            if diff < best_diff:
                best_diff = diff
                best_level = level
        return best_level

    # ------------------------------------------------------------------
    # Mask loading & object detection
    # ------------------------------------------------------------------

    def load_raw_mask(self, mask_path: Path, target_wh: Tuple[int, int]) -> np.ndarray:
        """Load greyscale mask and resize with nearest-neighbour (preserves labels)."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask from {mask_path}")
        return cv2.resize(mask, target_wh, interpolation=cv2.INTER_NEAREST)

    def detect_objects(self, mask: np.ndarray) -> List[int]:
        unique = np.unique(mask)
        return sorted(int(v) for v in unique if v != 0)

    def get_object_regions(
        self,
        mask: np.ndarray,
        slide_level_dims: Tuple[int, int],
    ) -> List[ObjectRegion]:
        slide_w, slide_h = slide_level_dims
        regions: List[ObjectRegion] = []
        for obj_id in self.detect_objects(mask):
            binary = (mask == obj_id).astype(np.uint8)
            ys, xs = np.where(binary)
            if len(xs) == 0:
                continue
            p = self.object_padding
            x_pad = max(0, int(xs.min()) - p)
            y_pad = max(0, int(ys.min()) - p)
            x_end = min(slide_w, int(xs.max()) + p + 1)
            y_end = min(slide_h, int(ys.max()) + p + 1)
            regions.append(ObjectRegion(
                object_id=obj_id,
                x=x_pad, y=y_pad,
                w=x_end - x_pad, h=y_end - y_pad,
                binary_mask=binary[y_pad:y_end, x_pad:x_end],
            ))
        return regions

    # ------------------------------------------------------------------
    # Coordinate generation
    # ------------------------------------------------------------------

    def generate_patch_coordinates_for_region(
        self,
        region: ObjectRegion,
        downsample: float,
    ) -> List[Dict]:
        stride = self.patch_size - self.overlap
        coords: List[Dict] = []
        for y in range(0, region.h - self.patch_size, stride):
            for x in range(0, region.w - self.patch_size, stride):
                patch_mask = region.binary_mask[y:y + self.patch_size,
                                                x:x + self.patch_size]
                if patch_mask.shape != (self.patch_size, self.patch_size):
                    continue
                if np.mean(patch_mask) < self.tissue_threshold:
                    continue
                abs_x = region.x + x
                abs_y = region.y + y
                coords.append({
                    'level_x': abs_x,
                    'level_y': abs_y,
                    'slide_x': int(abs_x * downsample),
                    'slide_y': int(abs_y * downsample),
                })
        return coords

    # ------------------------------------------------------------------
    # Threaded patch reading
    # ------------------------------------------------------------------

    def read_patches_threaded(
        self,
        svs_path: Path,
        coords: List[Dict],
        level: int,
    ) -> np.ndarray:
        """
        Read all patches for one object using a thread pool.

        OpenSlide is thread-safe for reads.  Each thread opens its own
        slide handle to avoid contention on the internal cache lock.

        Returns an array of shape (N, patch_size, patch_size, 3).
        """
        n = len(coords)
        patches = np.empty((n, self.patch_size, self.patch_size, 3), dtype=np.uint8)

        # Map (slide_x, slide_y) -> index so results land in the right slot
        coord_index = {(c['slide_x'], c['slide_y']): i for i, c in enumerate(coords)}

        with ThreadPoolExecutor(max_workers=self.patch_read_workers) as executor:
            futures = {
                executor.submit(
                    _read_patch_worker,
                    str(svs_path),
                    c['slide_x'], c['slide_y'],
                    level,
                    self.patch_size,
                ): i
                for i, c in enumerate(coords)
            }
            for future in tqdm(
                as_completed(futures),
                total=n,
                desc="  reading patches",
                leave=False,
            ):
                sx, sy, patch = future.result()
                patches[coord_index[(sx, sy)]] = patch

        return patches

    # ------------------------------------------------------------------
    # HDF5 saving (serial - h5py is not thread-safe for writes)
    # ------------------------------------------------------------------

    def _save_object_patches(
        self,
        patches: np.ndarray,
        coords: List[Dict],
        region: ObjectRegion,
        output_path: Path,
    ) -> int:
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('patches', data=patches, compression='gzip')
            f.create_dataset(
                'level_coordinates',
                data=np.array([[c['level_x'], c['level_y']] for c in coords], dtype=np.int32),
            )
            f.create_dataset(
                'slide_coordinates',
                data=np.array([[c['slide_x'], c['slide_y']] for c in coords], dtype=np.int32),
            )
            f.attrs['object_id'] = region.object_id
            f.attrs['bbox_x']    = region.x
            f.attrs['bbox_y']    = region.y
            f.attrs['bbox_w']    = region.w
            f.attrs['bbox_h']    = region.h
        return len(coords)

    # ------------------------------------------------------------------
    # Single-slide processing
    # ------------------------------------------------------------------

    def process_single_slide(
        self,
        svs_path: Path,
        mask_path: Path,
        output_dir: Path,
        slide_id: str,
    ) -> List[Dict]:
        """
        Process one WSI.  Returns a list of result dicts (one per object).
        Safe to call from a worker process.
        """
        slide = openslide.open_slide(str(svs_path))
        level      = self.get_magnification_level(slide)
        level_dims = slide.level_dimensions[level]   # (W, H)
        downsample = slide.level_downsamples[level]
        slide.close()   # Close early; re-opened per-thread inside reader

        mask    = self.load_raw_mask(mask_path, level_dims)
        objects = self.detect_objects(mask)
        multi   = len(objects) > 1
        print(f"[{slide_id}] {len(objects)} object(s): {objects}")

        slide_dir = output_dir / str(slide_id)
        slide_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict] = []
        for region in self.get_object_regions(mask, level_dims):
            coords = self.generate_patch_coordinates_for_region(region, downsample)
            if not coords:
                print(f"  [{slide_id}] obj {region.object_id}: 0 patches - skipping")
                continue

            suffix  = f"_obj{region.object_id:03d}" if multi else ""
            h5_path = slide_dir / f"{slide_id}{suffix}_patches.h5"

            print(
                f"  [{slide_id}] obj {region.object_id}: "
                f"bbox=({region.x},{region.y},{region.w}x{region.h}), "
                f"patches={len(coords)}"
            )

            # Step 1: Read patches in parallel using thread pool
            patches = self.read_patches_threaded(svs_path, coords, level)

            # Step 2: Write to HDF5 serially (h5py not thread-safe)
            n = self._save_object_patches(patches, coords, region, h5_path)

            results.append({
                'Image':          slide_id,
                'object_id':      region.object_id,
                'num_patches':    n,
                'h5_path':        str(h5_path),
                'level':          level,
                'magnification':  self.target_mag,
            })

        return results

    # ------------------------------------------------------------------
    # Dataset-level processing (parallel across slides)
    # ------------------------------------------------------------------

    def process_dataset(
        self,
        metadata_df: pd.DataFrame,
        svs_dir: Path,
        mask_dir: Path,
        output_dir: Path,
        slide_workers: int = 0,
    ) -> pd.DataFrame:
        """
        Process an entire dataset with slide-level parallelism.

        Each slide runs in its own subprocess (ProcessPoolExecutor).
        Within each subprocess, patch reads use a thread pool.

        Parameters
        ----------
        slide_workers : int
            Number of parallel slide processes.
            Rule of thumb: total_cores // patch_read_workers.
            E.g. 190 cores, 16 patch threads -> ~11 parallel slides.
            0 = auto.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if slide_workers <= 0:
            slide_workers = max(1, (os.cpu_count() or 4) // self.patch_read_workers)

        print(
            f"Parallelism: {slide_workers} slide process(es) x "
            f"{self.patch_read_workers} patch-read thread(s) each"
        )

        worker_args = []
        for _, row in metadata_df.iterrows():
            slide_id  = row['Image']
            svs_path  = svs_dir  / f"{slide_id}.svs"
            mask_path = mask_dir / f"{slide_id}_mask.png"

            if not svs_path.exists():
                print(f"Warning: SVS not found: {svs_path}")
                continue
            if not mask_path.exists():
                print(f"Warning: Mask not found: {mask_path}")
                continue

            worker_args.append({
                'svs_path':           str(svs_path),
                'mask_path':          str(mask_path),
                'output_dir':         str(output_dir),
                'slide_id':           slide_id,
                'patch_size':         self.patch_size,
                'target_mag':         self.target_mag,
                'overlap':            self.overlap,
                'tissue_threshold':   self.tissue_threshold,
                'object_padding':     self.object_padding,
                'patch_read_workers': self.patch_read_workers,
            })

        all_results: List[Dict] = []

        # Use 'spawn' start method to avoid OpenSlide fork-safety issues on Linux
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=slide_workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(_process_slide_worker, args): args['slide_id']
                for args in worker_args
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Slides"):
                slide_id = futures[future]
                try:
                    all_results.extend(future.result())
                except Exception as exc:
                    print(f"ERROR [{slide_id}]: {exc}")

        results_df = pd.DataFrame(all_results)
        output_df  = metadata_df.merge(results_df, on='Image', how='left')
        output_df.to_csv(output_dir / 'processed_metadata.csv', index=False)
        return output_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="WSI patch extraction")
    parser.add_argument('--metadata_csv',       type=Path,  default=Path("metadata.csv"))
    parser.add_argument('--svs_dir',            type=Path,  default=Path("/path/to/svs"))
    parser.add_argument('--mask_dir',           type=Path,  default=Path("/path/to/masks"))
    parser.add_argument('--output_dir',         type=Path,  default=Path("processed_patches"))
    parser.add_argument('--patch_size',         type=int,   default=256)
    parser.add_argument('--magnification',      type=float, default=20.0)
    parser.add_argument('--tissue_threshold',   type=float, default=0.5)
    parser.add_argument('--object_padding',     type=int,   default=64)
    parser.add_argument(
        '--slide_workers', type=int, default=0,
        help="Parallel slide processes. 0=auto (cpu_count // patch_read_workers)."
    )
    parser.add_argument(
        '--patch_read_workers', type=int, default=16,
        help="Threads per slide process for concurrent patch reading."
    )
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata_csv)

    preprocessor = WSIPreprocessor(
        patch_size=args.patch_size,
        target_magnification=args.magnification,
        tissue_threshold=args.tissue_threshold,
        object_padding=args.object_padding,
        patch_read_workers=args.patch_read_workers,
    )

    processed_df = preprocessor.process_dataset(
        metadata_df,
        args.svs_dir,
        args.mask_dir,
        args.output_dir,
        slide_workers=args.slide_workers,
    )

    print(f"\nSlides processed : {processed_df['Image'].nunique()}")
    print(f"Objects found    : {len(processed_df)}")
    print(f"Total patches    : {processed_df['num_patches'].sum()}")


if __name__ == "__main__":
    main()
