# To Dos

1. Create 4096x4096 non-overlapping tiles of all images using the tissue outlines as a bounding box to minimize slide background
   1. Using QuPath to generate the tiles automatically, may need to exclude tiles that are not 4096x4096?
2. Create a subset of 296x296 non-overlapping tiles from all tiles in step one
3. Use these tiles for pre-training the ViT models for their respective tile sizes