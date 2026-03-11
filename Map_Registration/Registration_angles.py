import numpy as np
import SimpleITK as sitk
from typing import Tuple
import skimage as sk
from PIL import Image
from scipy import ndimage
import pandas as pd
import os
import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt
import math

def process_data(animal_id, tile_centroids, anno_data, grey_value_df, map_path, data_path):
    try:
        anno_df = anno_data[anno_data['AnimalID'] == animal_id]
        glands = anno_df['Gland.side'].unique().tolist()
        for gland in glands:
            try:        
                gland_df = anno_df[anno_df['Gland.side'] == gland]
                map_base = gland_df['MapBase'].unique().tolist()[0]
                map_base_path = os.path.join(map_path, f'{map_base}.png')
                map_arr = np.array(Image.open(map_base_path).convert('L'))
                tile_centroid_df_animal = tile_centroids[tile_centroids['Tiles_AnimalID'] == animal_id]
                tile_centroid_df_gland = tile_centroid_df_animal[tile_centroid_df_animal['Tiles_Gland_side'] == gland]
                spacing = (16.1, 16.1)
                reg = Registration.__new__(Registration)
                saved_df_paths = []
                transformed_tile_dfs = []
                map_region_failures = []
                orientation_failures = []  # Bug 1 fixed
                cardinal_point_failures = []
                refine_transform_failures = []

                for _, row in gland_df.iterrows():
                    name = row['Image'] + '_' + row['Tissue.ID']
                    try:                                          # outer row try
                        tissue_path = os.path.join(data_path, row['Tissue.ID'], row['Image'] + '.svs.png')
                        tissue_arr = reg.open_img(tissue_path)
                        img_bbox, cropped_mask = reg.get_tissue_bbox(tissue_arr, row['Image'], row['Tissue.ID'], data_path)
                        pad_img_bbox, pad_mask_bbox = reg.add_padding(img_bbox, cropped_mask, row['Image'], row['Tissue.ID'], data_path)
                        try:
                            points = reg.get_cardinal_points(pad_img_bbox, name, grey_value_df, ('north', 'east'))
                        except ValueError as e:
                            print(f"  Cardinal Point Detection failure logged for {name}")
                            cardinal_point_failures.append({
                                'Image': row['Image'],
                                'Tissue': row['Tissue.ID']
                            })
                            continue

                        try:                                      # orientation try
                            flip, angle_radians, angle_degrees = reg.orient_tissue(points, name, grey_value_df, pad_mask_bbox)
                        except ValueError as e:
                            if "ORIENTATION_MISMATCH" in str(e):
                                print(f"  Orientation failure logged for {name}")
                                orientation_failures.append({
                                    'Image': row['Image'],
                                    'Tissue_ID': row['Tissue.ID']
                                })
                                continue
                            raise

                        try:                                      # map region try
                            map_region = reg.get_map_region(map_arr, row['MappingID'], grey_value_df)
                        except ValueError:
                            print(f"  Map region failure logged for {name}")
                            map_region_failures.append({
                                'Image': row['Image'],
                                'Mapping_ID': row['MappingID'],
                                'Map_Base': map_base
                            })
                            continue

                        sitk_fixed, sitk_moving = reg.load_sitk_imgs(map_region, pad_mask_bbox, spacing)
                        composite_transform = reg.apply_flip_rotation(sitk_moving, sitk_fixed, flip, angle_radians, angle_degrees)
                        try:
                            transform = reg.refine_registration(sitk_moving, sitk_fixed, composite_transform,
                                                            data_path, row['Tissue.ID'], row['Image'])
                        except ValueError:
                            print(f"  Refined Transform error logged for {name}")
                            refine_transform_failures.append({
                                'Image': row['Image'],
                                'Mapping_ID': row['MappingID'],
                                'Map_Base': map_base
                            })
                            continue

                        tile_coordinates = tile_centroid_df_gland[
                            (tile_centroid_df_gland['Tiles_Image'] == row['Image']) &
                            (tile_centroid_df_gland['Tiles_Parent'] == row['Tissue.ID'])
                        ]
                        df_path, transformed_tile_df = reg.transform_points(transform, tile_coordinates, data_path, name)
                        saved_df_paths.append(df_path)
                        transformed_tile_dfs.append(transformed_tile_df)
                        reg.plot_registered_tissue(padded_img=pad_img_bbox,map_region=map_region,spacing=spacing,transform=transform,name=name,path_to_tissue_masks=data_path)
                    except Exception as e:                        #outer row except
                        raise RuntimeError(f"Failed processing image {name}: {e}")
                if saved_df_paths:
                    concat_dfs = pd.concat(transformed_tile_dfs,ignore_index=True)
                    save_path_concat_dfs = os.path.join(data_path,'Transformed_Coordinates_per_Animal')
                    os.makedirs(save_path_concat_dfs,exist_ok=True)
                    concat_dfs.to_csv(os.path.join(save_path_concat_dfs,f'{animal_id}_{gland}_Transformed_Tile_Data.csv'),index=False)
                    reg.plot_transformed_points(saved_df_paths,animal_id,gland,data_path)
                reg.write_qc_logs(map_region_failures, orientation_failures, cardinal_point_failures, data_path, animal_id, gland)
            except Exception as e:
                print(f"Failed processing gland {gland} for animal {animal_id}: {e}")
                continue

    except Exception as e:
        raise RuntimeError(f"Failed processing animal {animal_id}: {e}")

class Registration:
    """ 
    Register each tissue mask to its corresponding map region and then use the transformation matrix
    to transform all X,Y coordinates of tile centroids from the same tissue in df.
    """
    
    def __init__(
            self,
            path_to_tissue_masks: str,
            path_to_maps: str,
            path_to_grey_value_key: str,
            path_to_tile_df: str,
            path_to_anno_df: str,
            genotype: str
    ):
        self.path_to_tissue_masks = path_to_tissue_masks
        self.path_to_maps = path_to_maps
        self.path_to_grey_value_key = path_to_grey_value_key
        self.path_to_tile_df = path_to_tile_df
        self.path_to_anno_df = path_to_anno_df
        self.genotype = genotype if genotype != None else 'WT'

    def set_up_pipeline(
            self,
            path_to_tissue_masks,
            path_to_maps,
            path_to_grey_value_key,
            path_to_tile_df,
            path_to_anno_df,
            genotype
    ):
        """
        Set up paths for parent location of tissue masks and maps
        Read in select columns from data frames for tile data, annotation data, and grey value key

        Args:
            path_to_tissue_masks
            path_to_maps
            path_to_grey_value_key
            path_to_tile_df
            path_to_anno_df
        Returns:
            data_path: Path
            map_path: Path
            tile_df: pd.df
            anno_df: pd.df
            grey_value_df: pd.df
        """
        data_path = Path(path_to_tissue_masks)
        map_path = Path(path_to_maps)
        tile_df = pd.read_csv(Path(path_to_tile_df),dtype=str,low_memory=False)
        tile_df['Tiles_Centroid_X_um'] = tile_df['Tiles_Centroid_X_um'].astype(float) 
        tile_df['Tiles_Centroid_Y_um'] = tile_df['Tiles_Centroid_Y_um'].astype(float)
        anno_df = pd.read_csv(Path(path_to_anno_df),dtype=str)
        grey_value_df = pd.read_csv(Path(path_to_grey_value_key),dtype=str,usecols=['Mapping_ID','Map_Grey_value','Tissue_Grey_value'])
        grey_value_df['Map_Grey_value'] = grey_value_df['Map_Grey_value'].astype(float)
        grey_value_df['Tissue_Grey_value'] = grey_value_df['Tissue_Grey_value'].astype(float)
        filtered_tile_df = tile_df[tile_df['Tiles_Genotype']==genotype]
        filtered_anno_df = anno_df[anno_df['Genotype']==genotype]
        del tile_df, anno_df
        return data_path, map_path, filtered_tile_df, filtered_anno_df, grey_value_df
        
    def write_qc_logs(self, map_region_failures, orientation_failures, cardinal_point_failures, refine_transform_failures, data_path, animal_id, gland):
        qc_path = os.path.join(data_path, 'QC_logs')
        os.makedirs(qc_path, exist_ok=True)
    
        if map_region_failures:
            map_df = pd.DataFrame(map_region_failures)
            map_df.to_csv(os.path.join(qc_path, f'{animal_id}_{gland}_map_region_failures.csv'), index=False)
            print(f"  {len(map_region_failures)} map region failure(s) written for {animal_id} {gland}")
    
        if orientation_failures:
            orient_df = pd.DataFrame(orientation_failures)
            orient_df.to_csv(os.path.join(qc_path, f'{animal_id}_{gland}_orientation_failures.csv'), index=False)
            print(f"  {len(orientation_failures)} orientation failure(s) written for {animal_id} {gland}")
            
        if cardinal_point_failures:
            cardinal_df = pd.DataFrame(cardinal_point_failures)
            cardinal_df.to_csv(os.path.join(qc_path, f'{animal_id}_{gland}_cardinal_point_failures.csv'), index=False)
            print(f"  {len(cardinal_point_failures)} cardinal point detection failure(s) written for {animal_id} {gland}")
        
        if refine_transform_failures:
            transform_df = pd.DataFrame(refine_transform_failures)
            transform_df.to_csv(os.path.join(qc_path, f'{animal_id}_{gland}_refine_transform_failures.csv'), index=False)
            print(f"  {len(refine_transform_failures)} cardinal point detection failure(s) written for {animal_id} {gland}")
    
        if not map_region_failures and not orientation_failures and not cardinal_point_failures and not refine_transform_failures:
            print(f"  No QC failures for {animal_id} {gland}")
        
    def get_animal_ids(
            self,
            filtered_anno_df
    ) -> list[str]:
        animal_ids = filtered_anno_df['AnimalID'].unique().tolist()
        return animal_ids
    
    def open_img(
            self,
            img_path: str
        ):
        arr = np.array(Image.open(img_path).convert('L'))
        return arr

    def get_tissue_bbox(
            self,
            tissue_arr,
            name,
            tissue_id,
            data_path
    ):
        props = sk.measure.regionprops(tissue_arr)
        largest_obj = max(props, key=lambda p: p.area)
        largest_obj_label = largest_obj.label
        minc = min(props, key=lambda p: p.bbox[0])
        minr = min(props, key=lambda p: p.bbox[1])
        maxc = max(props, key=lambda p: p.bbox[2])
        maxr = max(props, key=lambda p: p.bbox[3])
        img_bbox = tissue_arr[minc.bbox[0]-10:maxc.bbox[2]+10,minr.bbox[1]-10:maxr.bbox[3]+10]
        cropped_mask = (img_bbox==largest_obj_label)*img_bbox
        cropped_mask = ndimage.binary_fill_holes(cropped_mask)
        save_path_img = os.path.join(data_path,tissue_id,'cropped_image')
        save_path_mask = os.path.join(data_path,tissue_id,'cropped_mask')
        os.makedirs(save_path_img,exist_ok=True)
        os.makedirs(save_path_mask,exist_ok=True)
        sk.io.imsave(os.path.join(save_path_img,'cropped_'+name+'.png'),img_bbox,check_contrast=False)
        sk.io.imsave(os.path.join(save_path_mask,'cropped_mask_'+name+'.png'),cropped_mask,check_contrast=False)
        return img_bbox, cropped_mask
    
    def add_padding(
            self,
            img_bbox,
            cropped_mask,
            name,
            tissue_id,
            data_path):
        """ 
        reshape the cropped images to a square based on the largest dim using padding
        saves new arrays with added padding of a constant value: 255 for grey scale image, 0 for masks
        returns save path for later use
        """
        max_dim = max(img_bbox.shape)
        padding_y = (max_dim - img_bbox.shape[0]) // 2
        padding_x = (max_dim - img_bbox.shape[1]) // 2
        pad_img_bbox = np.pad(img_bbox,((padding_y+50,padding_y+50),(padding_x+50,padding_x+50)),mode='constant',constant_values=255)
        pad_mask_bbox = np.pad(cropped_mask,((padding_y+50,padding_y+50),(padding_x+50,padding_x+50)),mode='constant',constant_values=0)
        save_path_img = os.path.join(data_path,tissue_id,'padded_cropped_image')
        save_path_mask = os.path.join(data_path,tissue_id,'padded_cropped_mask')
        os.makedirs(save_path_img,exist_ok=True)
        os.makedirs(save_path_mask,exist_ok=True)
        sk.io.imsave(os.path.join(save_path_img,'padded_cropped_'+name+'.png'),pad_img_bbox,check_contrast=False)
        sk.io.imsave(os.path.join(save_path_mask,'padded_cropped_mask_'+name+'.png'),pad_mask_bbox,check_contrast=False)
        return pad_img_bbox,pad_mask_bbox
    
    def get_map_region(
            self,
            map_arr,
            map_id,
            grey_value_df):
        grey_value = grey_value_df.loc[grey_value_df['Mapping_ID'] == map_id, 'Map_Grey_value'].values[0] 
        map_region = (map_arr == grey_value).astype(np.float32)
        map_region = ndimage.binary_fill_holes(map_region)
        labels = sk.measure.label(map_region)
        props = sk.measure.regionprops(labels)
        largest_obj = max(props, key=lambda p: p.area)
        largest_obj_label = largest_obj.label
        masked_map_region = (labels==largest_obj_label)*map_region
        if not map_region.any():
            raise ValueError(f"Map region for ID '{map_id}' (grey value {grey_value}) is empty check map image and grey value key.")
        return masked_map_region
    
    def detect_cardinal_point(
            self,
            img_bbox, 
            grey_value, 
            name):
        """
        Detect the centroid of a cardinal point marker by its grey value.
        
        Returns (x, y) in pixel coordinates, or None if not found.
        """
        
        # Create binary mask for this grey value.
        # Use a small tolerance instead of exact equality to avoid failures when images
        # were saved with lossy settings or resampled previously.
        gv = float(grey_value)
        mask = np.abs(img_bbox.astype(np.float32) - gv) <= 1.0
            
        if not mask.any():
            print(f"  WARNING: No pixels found for grey value {grey_value} in {name}")
            return None
        
        # Label connected components and get centroid
        labeled_img = sk.measure.label(mask.astype(np.uint8))
        props = sk.measure.regionprops(labeled_img)
        if not props:
            print(f"  WARNING: No connected components found for grey value {grey_value} in {name}")
            return None

        largest = max(props, key=lambda p: p.area)
        cy, cx = largest.centroid
        return np.array([cy, cx], dtype=float)  # return as (y,x)
    
    def get_cardinal_points(
            self,
            img_bbox,
            name,
            grey_value_df,
            direction):
        """
        Extract one or more cardinal points from an image.

        Parameters
        ----------
        direction:
            Either a single direction string (e.g., 'north') or an iterable of
            directions (e.g., ('north', 'east')).

        Returns
        -------
        dict[str, np.ndarray]
            Keys are directions and values are (y,x) arrays (float).
        """
        points = {}
        directions = (direction,) if isinstance(direction, str) else tuple(direction)
        for d in directions:
            grey = grey_value_df.loc[grey_value_df['Mapping_ID'] == d, 'Tissue_Grey_value'].values[0]
            print(f'Grey value for {d} is {grey}')
            pt = self.detect_cardinal_point(img_bbox, grey, name)
            if pt is None:
                raise ValueError(f"Could not find '{d}' point in {name}")
            points[d] = pt
            print(f"  {d}: pixel (y,x) ({pt[0]:.1f}, {pt[1]:.1f})")
        return points
    
    def orient_tissue(
            self,
            points,
            name,
            mask_arr):
        """
        Determine the flip and/or rotation needed to orient the moving image so that
        north points up and east points left, with "true north" defined as orthogonal
        to the detected east-west axis.

        Parameters
        ----------
        points:
            dict with 'north' and 'east' as (y,x) pixel arrays in the moving image.
        mask_arr:
            Tissue mask array used to compute the tissue centroid (non-zero pixels treated as tissue).
        """
        if 'north' not in points or 'east' not in points:
            raise ValueError(f"Missing required cardinal points for {name}. Found: {list(points.keys())}")

        tissue_pixels = np.argwhere(np.asarray(mask_arr) > 0)
        if tissue_pixels.size == 0:
            raise ValueError(f"No tissue mask pixels found for {name}")
        cy, cx = tissue_pixels.mean(axis=0)  # (y, x)

        def _wrap_degrees(deg: float) -> float:
            # Map to [-180, 180)
            return ((deg + 180.0) % 360.0) - 180.0

        # Work in vector form around the tissue centroid to avoid dependence on image center.
        # Use EAST to define the true east-west axis, then choose the 180° branch that makes NORTH point up.
        y_n, x_n = points['north']
        y_e, x_e = points['east']
        dx_n, dy_n = (x_n - cx), (y_n - cy)
        dx_e, dy_e = (x_e - cx), (y_e - cy)

        # CCW rotation (screen/image coordinates) needed to rotate the EAST vector to point LEFT (-x).
        # angle_e is the current angle of EAST relative to +x; desired is 180° (left).
        angle_e = math.degrees(math.atan2(dy_e, dx_e))
        angle_degrees = _wrap_degrees(180.0 - angle_e)

        def _rotate(dx: float, dy: float, deg: float) -> tuple[float, float]:
            r = math.radians(deg)
            c = math.cos(r)
            s = math.sin(r)
            return (c * dx - s * dy, s * dx + c * dy)

        # Ensure NORTH ends up pointing up (-y). If not, rotate an additional 180°.
        _, dy_n_rot = _rotate(dx_n, dy_n, angle_degrees)
        if dy_n_rot > 0:
            angle_degrees = _wrap_degrees(angle_degrees + 180.0)

        # Determine horizontal flip after rotation to enforce EAST to the left.
        dx_e_rot, _ = _rotate(dx_e, dy_e, angle_degrees)
        if abs(dx_e_rot) < 1.0:
            print('East is on (or too close to) the midline after rotation, check image')
            raise ValueError("ORIENTATION_MISMATCH")
        flip = 'horizontal' if dx_e_rot > 0 else None

        rads = math.radians(angle_degrees)
        print(f'Detected Flip: {flip}')
        print(f'Detected Rotation: {angle_degrees:.2f} degrees ({rads:.4f} radians)')
        return flip, rads, angle_degrees

    def load_sitk_imgs(self,map_region,pad_mask_bbox,spacing):
        """ Convert array to 32bit float and then to sitk image for registration """
        bitdepth_map = np.array(map_region).astype(np.float32)
        sitk_fixed = sitk.GetImageFromArray(bitdepth_map)
        sitk_fixed.SetSpacing(spacing)
        bitdepth_mask = np.array(pad_mask_bbox).astype(np.float32)
        sitk_moving = sitk.GetImageFromArray(bitdepth_mask)
        sitk_moving.SetSpacing(spacing)
        return sitk_fixed, sitk_moving

    def get_mask_centroid(self,sitk_image):
        """Extract physical centroid of the mask region."""
        binary = sitk.BinaryThreshold(
            sitk_image,
            lowerThreshold= 0.5,
            upperThreshold= 255.0,
            insideValue=1,
            outsideValue=0
        )
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(binary)
        if 1 not in label_stats.GetLabels():
            raise ValueError("No mask found — check your threshold.")
        return label_stats.GetCentroid(1)  # returns physical coords

    def get_centroid_alignment_transform(self, sitk_moving, sitk_fixed):
        """
        Build a translation transform that maps the fixed mask centroid to the
        moving mask centroid.  SimpleITK resampling uses the transform in the
        fixed→moving direction: T(p) = p + offset, so T(fixed_centroid) =
        moving_centroid requires offset = moving_centroid - fixed_centroid.
        """
        fixed_centroid  = self.get_mask_centroid(sitk_fixed)
        moving_centroid = self.get_mask_centroid(sitk_moving)

        translation = sitk.TranslationTransform(sitk_moving.GetDimension())
        offset = [m - f for f, m in zip(fixed_centroid, moving_centroid)]
        translation.SetOffset(offset)
        return translation

    def apply_flip_rotation(self, sitk_moving, sitk_fixed, flip=None, rads=None, angle_degrees=None):
        """
        Build a composite pre-alignment transform (fixed→moving, backward convention).

        Forward orientation operations applied to the moving image:
          1. Rotate CCW by angle_degrees around moving mask centroid → north to top
          2. Flip horizontally around moving mask centroid (if needed) → east to left
          3. Translate moving centroid to fixed centroid

        Backward (fixed→moving) composite undoes these in reverse order.
        SimpleITK CompositeTransform applies transforms last-added → first-applied,
        so we add transforms in reverse application order:
          add 1st (applied last): rotation by -rads  (un-rotate)
          add 2nd (applied 2nd):  horizontal flip    (un-flip, self-inverse)
          add 3rd (applied 1st):  centroid alignment (un-translate)

        All rotation and flip transforms are centred on the physical centroid of the
        moving mask (not the image centre) so that the centroid stays fixed under
        rotation and flip, guaranteeing the aligned centroid remains within the
        moving image bounds.
        """
        composite_transform = sitk.CompositeTransform(2)

        # Centroid of the moving mask in physical coordinates — used as the centre
        # for both rotation and flip so that those transforms leave the centroid fixed.
        moving_centroid = self.get_mask_centroid(sitk_moving)

        if rads is not None:
            print(f"  Applying rotation: {angle_degrees:.2f} degrees")
            rotation_transform = sitk.Euler2DTransform()
            rotation_transform.SetCenter(moving_centroid)
            rotation_transform.SetAngle(-rads)
            composite_transform.AddTransform(rotation_transform)

        if flip is not None:
            print(f"  Applying flip: {flip}")
            flip_transform = sitk.AffineTransform(sitk_moving.GetDimension())
            flip_transform.SetCenter(moving_centroid)
            flip_transform.SetMatrix([-1, 0,  0,  1])   # horizontal flip: x → -(x-cx)+cx
            composite_transform.AddTransform(flip_transform)

        # Centroid alignment: translate fixed centroid → moving centroid.
        centroid_transform = self.get_centroid_alignment_transform(sitk_moving, sitk_fixed)
        composite_transform.AddTransform(centroid_transform)

        if flip is None and rads is None:
            print("  No flip or rotation applied.")

        return composite_transform
    
    def to_distance_map(self, sitk_img):
        """Convert binary mask to signed Maurer distance map for smoother optimization."""
        binary = sitk.BinaryThreshold(sitk_img, lowerThreshold=0.5, upperThreshold=255.0,
                                    insideValue=1, outsideValue=0)
        return sitk.SignedMaurerDistanceMap(
            sitk.Cast(binary, sitk.sitkUInt8),
            insideIsPositive=True,
            squaredDistance=False,
            useImageSpacing=True
        )
    
    def refine_registration(self,
                            sitk_moving,
                            sitk_fixed,
                            composite_transform,
                            data_path,
                            tissue_id,
                            name,
                            max_rotation_degrees: float = 0.0,
                            rotation_step_degrees: float = 2.0):
        save_path_hdf = os.path.join(data_path,tissue_id,'transformation_files','hdf')
        save_file_hdf = os.path.join(save_path_hdf,f"TF_hdf_{name}_{tissue_id}.hdf")
        os.makedirs(save_path_hdf,exist_ok=True)

        moving_dist = self.to_distance_map(sitk_moving)
        fixed_dist = self.to_distance_map(sitk_fixed)

        fixed_mask_sitk = sitk.BinaryThreshold(
            sitk_fixed,
            lowerThreshold=0.5,
            upperThreshold=255.0,
            insideValue=1,
            outsideValue=0
        )

        def _run_translation_refine(
                moving_initial_transform,
                iterations: int,
                learning_rate: float = 1.0
        ) -> tuple[float, sitk.Transform]:
            # Keep the north-up / east-left orientation fixed during optimization by
            # only refining translation (no additional rotation/scale/shear/reflection).
            fine = sitk.TranslationTransform(2)
            method = sitk.ImageRegistrationMethod()
            method.SetMetricFixedMask(fixed_mask_sitk)
            method.SetMetricSamplingStrategy(method.RANDOM)
            method.SetMetricSamplingPercentage(0.2)
            method.SetMetricAsJointHistogramMutualInformation()
            method.MetricUseFixedImageGradientFilterOff()
            method.SetMovingInitialTransform(moving_initial_transform)
            method.SetInitialTransform(fine, inPlace=True)
            method.SetOptimizerAsGradientDescent(
                learningRate=learning_rate,
                numberOfIterations=int(iterations),
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=20
            )
            method.SetOptimizerScalesFromPhysicalShift()
            method.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
            method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
            method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            method.SetInterpolator(sitk.sitkLinear)
            method.Execute(fixed_dist, moving_dist)
            return float(method.GetMetricValue()), fine

        # Optional bounded rotation refinement: search over a limited range of extra
        # rotation (+/- max_rotation_degrees) and keep the best MI score after a short
        # translation-only refinement. This enforces a hard cap on rotation drift.
        chosen_initial = composite_transform
        chosen_delta_deg = 0.0
        if max_rotation_degrees and max_rotation_degrees > 0:
            step = float(rotation_step_degrees) if rotation_step_degrees and rotation_step_degrees > 0 else 2.0
            max_deg = float(max_rotation_degrees)
            # Coarse-to-fine search.
            coarse_angles = np.arange(-max_deg, max_deg + (step * 0.5), step, dtype=float)
            moving_centroid = self.get_mask_centroid(sitk_moving)

            def _with_delta_rotation(delta_deg: float):
                delta = sitk.Euler2DTransform()
                delta.SetCenter(moving_centroid)
                delta.SetAngle(math.radians(float(delta_deg)))
                # Applied after the provided composite_transform (rotate in moving-space about centroid).
                out = sitk.CompositeTransform(2)
                out.AddTransform(delta)               # applied last
                out.AddTransform(composite_transform) # applied first
                return out

            best_metric = None
            best_deg = 0.0
            for deg in coarse_angles:
                candidate_initial = _with_delta_rotation(deg)
                metric, _ = _run_translation_refine(candidate_initial, iterations=80, learning_rate=1.0)
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_deg = float(deg)

            fine_step = max(step / 4.0, 0.25)
            fine_angles = np.arange(best_deg - step, best_deg + step + (fine_step * 0.5), fine_step, dtype=float)
            for deg in fine_angles:
                if abs(deg) > max_deg + 1e-6:
                    continue
                candidate_initial = _with_delta_rotation(deg)
                metric, _ = _run_translation_refine(candidate_initial, iterations=120, learning_rate=0.5)
                if metric < best_metric:
                    best_metric = metric
                    best_deg = float(deg)

            chosen_delta_deg = best_deg
            chosen_initial = _with_delta_rotation(best_deg)
            print(f"  Rotation search: chose dtheta={chosen_delta_deg:.2f} deg within +/-{max_deg:.2f} deg")

        print(f"  Registering...")
        metric, fine_transform = _run_translation_refine(chosen_initial, iterations=500, learning_rate=1.0)
        print(f"metric: {metric:.6f}")

        full_composite = sitk.CompositeTransform(2)
        full_composite.AddTransform(chosen_initial)
        full_composite.AddTransform(fine_transform)

        #save transform as mat and csv
        full_composite.FlattenTransform()
        full_composite.WriteTransform(save_file_hdf)
        return full_composite
    
    def transform_points(self,transform,tile_df,path_to_tissue_masks,name):
        save_path = os.path.join(path_to_tissue_masks,'Transformed_Coordinates')
        os.makedirs(save_path,exist_ok=True)
        save_df = os.path.join(save_path,name+'_Transformed_Coordinates.csv')
        inverse_transform = transform.GetInverse()
        transformed = tile_df.apply(
            lambda row: inverse_transform.TransformPoint((row['Tiles_Centroid_X_um'], row['Tiles_Centroid_Y_um'])),
            axis=1)
        tile_df['Tiles_Transformed_X_um'] = transformed.apply(lambda p: p[0])
        tile_df['Tiles_Transformed_Y_um'] = transformed.apply(lambda p: p[1])
        tile_df.to_csv(save_df,index=False)
        return save_df,tile_df

    def plot_transformed_points(self,saved_dfs,animal_id,gland,data_path):
        dfs = []
        save_path = os.path.join(data_path,'Scatter_plots')
        save_plot = os.path.join(save_path,f"transformed_coordnates_{animal_id}_{gland}.png")
        os.makedirs(save_path,exist_ok=True)
        for df in saved_dfs:
            dfs.append(pd.read_csv(df,dtype=float,usecols=['Tiles_Transformed_X_um','Tiles_Transformed_Y_um']))
        color_list = ['b','g','m','c','y','k','r']
        fig, ax = plt.subplots()
        for i, df in enumerate(dfs):
            ax.scatter(df['Tiles_Transformed_X_um'],df['Tiles_Transformed_Y_um'],c=color_list[i % len(color_list)],alpha=0.5)
        ax.invert_yaxis()
        fig.savefig(save_plot, dpi=600)
        plt.close(fig)

    
    def plot_registered_tissue(self,padded_img,map_region,spacing,transform,name,path_to_tissue_masks):
        sitk_fixed, sitk_moving = self.load_sitk_imgs(map_region, padded_img, spacing)
        save_path = os.path.join(path_to_tissue_masks,'Registered_Tissue_Overlays')
        os.makedirs(save_path,exist_ok=True)
        save_img = os.path.join(save_path,name+'_Registered_Overlay.png')
        resampled = sitk.Resample(
                sitk_moving, sitk_fixed, transform,
                sitk.sitkNearestNeighbor, 0.0, sitk_moving.GetPixelID()
            )
        fig, axes = plt.subplots()
        fixed  = sitk.GetArrayFromImage(sitk_fixed)
        moved  = sitk.GetArrayFromImage(resampled)
        axes.imshow(fixed, cmap="gray")
        axes.imshow(moved,cmap="Reds",alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_img, dpi=600)
        plt.close()

    def run(self):
        data_path, map_path, filtered_tile_df, filtered_anno_df, grey_value_df = self.set_up_pipeline(self.path_to_tissue_masks,
                                                                                                        self.path_to_maps,
                                                                                                        self.path_to_grey_value_key,
                                                                                                        self.path_to_tile_df,
                                                                                                        self.path_to_anno_df,
                                                                                                        self.genotype)
        animal_ids = self.get_animal_ids(filtered_anno_df)
        start = time.time()
        run_summary = {"success":0,"failed":0}
        if not animal_ids:
            print("No items found.")
            return
        print(f"Running pipeline on {len(animal_ids)} animal datasets...\n")
        
        for animal_id in animal_ids:
            try:
                process_data(animal_id, filtered_tile_df, filtered_anno_df, grey_value_df, map_path, data_path)
                run_summary["success"] += 1
            except Exception as e:
                print(f" ERROR on animal {animal_id}: {e}")
                run_summary["failed"] += 1
        elapsed = time.time() - start
        print(f"\nCompleted in {elapsed:.2f}s")
        print(f"\nSummary: {run_summary}")


def parse_args():
    parser = argparse.ArgumentParser(description='Register tissue masks to corresponding map regions and transform tile centroids for respective tissue')
    parser.add_argument('--map_path', type=str, required=True, help='Path to resized maps (.png)')
    parser.add_argument('--tissue_mask_path', type=str, required=True, help='Path to parent folder of tissue masks (.png)')
    parser.add_argument('--grey_value_key', type=str, required=True, help='Path to grey value key (.csv)')
    parser.add_argument('--tile_centroid_dataframe', type=str, required=True, help='Path to data frame of tile centroids (.csv)')
    parser.add_argument('--anno_dataframe', type=str, required=True, help='Path to data frame of annotations (.csv)')
    parser.add_argument('--genotype', type=str, required=True, help='Genotype to process')
    return parser.parse_args()


def main():
    args = parse_args()
    process = Registration(path_to_tissue_masks=args.tissue_mask_path,
                           path_to_maps=args.map_path,
                           path_to_grey_value_key=args.grey_value_key,
                           path_to_tile_df=args.tile_centroid_dataframe,
                           path_to_anno_df=args.anno_dataframe,
                           genotype=args.genotype)
    process.run()

if __name__ == '__main__':
    main()
