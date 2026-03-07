import numpy as np
from typing import Tuple
import skimage as sk
from PIL import Image
import scipy as sc
from glob import glob
import os
from tqdm import tqdm
import argparse

class Resize_Maps:
    """ 
    Register each tissue mask to its corresponding map region and then use the transformation matrix
    to transform all X,Y coordinates of tile centroids from the same tissue in df.
    """
    
    def __init__(
            self,
            path_to_maps: str,
    ):
        self.path_to_maps = path_to_maps

    def calculate_avg_tissue_size(self):
        avg_y = round(float(np.average([5.3,6.1])),4)
        avg_x = round(float(np.average([4.1,4.3])),4)
        return avg_y, avg_x
    
    def load_map_arrs(
            self,
            path_to_maps
            ) -> Tuple[str,np.ndarray]:
        map_files = sorted(glob(os.path.join(path_to_maps,'*.png')))
        map_names = list(map(os.path.basename,map_files))
        map_imgs = [Image.open(map_file).convert('L') for map_file in map_files]
        map_arrays = list(map(np.array,map_imgs))
        print(f'Found {len(map_files)} to process')
        return map_names,map_arrays
    
    def resize_maps(self,
                map_arr,
                avg_y,
                avg_x,
                target_spacing = 16.1):
        """ 
        Resizes image based on target spacing
        Returns the resized image
        """
        map_mask = map_arr < 255
        map_labels = sk.measure.label(map_mask)
        props = sk.measure.regionprops(map_labels)
        map_region = max(props, key=lambda p: p.area)
        area_bbox = map_region.area_bbox
        avg_area = avg_x*avg_y
        spacing_um = np.sqrt(round((avg_area / area_bbox) * 10000**2, 4))
        zoom = spacing_um/target_spacing
        resized_map = sc.ndimage.zoom(map_arr, zoom,order=0)
        return resized_map
    
    def save_maps(self,
                  path_to_maps,
                  names,
                  map_arrays):
        save_path = os.path.join(path_to_maps,'resized_maps')
        os.makedirs(save_path,exist_ok=True)
        for name,map_array in zip(names,map_arrays):
            Image.fromarray(map_array).save(os.path.join(save_path,name))
            print(f'Saved {name}')

    def run(self):
        avg_y, avg_x = self.calculate_avg_tissue_size()
        map_names, map_arrays = self.load_map_arrs(self.path_to_maps)
        resized_maps = []
        for map_arr in tqdm(map_arrays):
           resized_map = self.resize_maps(map_arr,avg_y,avg_x)
           resized_maps.append(resized_map)
        self.save_maps(self.path_to_maps,map_names,resized_maps)

def parse_args():
    parser = argparse.ArgumentParser(description='Resize Maps to 16.1um/px spacing')
    parser.add_argument('--map_path', type=str, required=True, help='Path to original maps (.png)')
    return parser.parse_args()
  

def main():
    args = parse_args()
    resize_maps = Resize_Maps(path_to_maps=args.map_path)
    resize_maps.run()

if __name__ == '__main__':
    main()