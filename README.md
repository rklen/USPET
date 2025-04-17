# USPET

USPET is an automatic segmentation method designed for dynamic PET images. It is computationally light enough for analysing modern images with large field-of-view covering tens of millions of voxels.

USPET uses following packages: sys, scipy, numba, numpy, sklearn, cc3d, scipy, datetime. The main function call is
uspet(image, clusters=100, extract_foreground="auto", do_cc=True, min_size=30, pca=5, location_weight=0.2, do_log=True, method="gmm")

where the only mandatory argument image is a 4D array containing the original dynamic image. It can be loaded for example from a .nii file using package nibabel like this:

import nibabel as nib

image = nib.load("path/to/my/pet/image/pet.nii").get_fdata()
segments = uspet(image)

and the obtained segmentation can be saved in a nifti format like this:

### Arguments

| argument | explanation | 
| -------- | ----------- |
| image | 4D array, contains the actual dynamic image |
| clusters | integer, the number of segments to make (if do_cc is True, this will not be final) |
| extract_foreground | indicates which voxels to use for segmentation, should be either 
#                             3D array (mask corresponding to the physical dimensions of the image, 
#                             2D array (foreground voxels as rows, x, y, and z coordinates as cols), 
#                             "auto" (automatically assumes voxels with ), or
#                             False (all voxels are used, don't use this, it is super slow).  |
| do_cc | boolean, defines if the output clusters should be broken into connected components |
| min_size | integer, defines how small segments are allowed |
| pca | integer (or None), the number of principal components |
| location_weight | float, a value between (inclusive) 0 and 1 indicating how much weight to put on location vs similarity of TACs. Value 0 means that location is not considered at all. |
| do_log | boolean, defines if log transformation should be made (usually a good idea) |
| method | string,  "gmm" or "k-means" defines the actual clustering method |

### Citation
If you use USPET in your research, please cite the original paper: FILL WHEN THE MANUSCRIPT IS ACCEPTED
