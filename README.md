# USPET

USPET is an automatic segmentation method designed for dynamic PET images. It is computationally light enough for analysing modern images with large field-of-view covering tens of millions of voxels. As a robust unsupervised general-purpose method, it can be applied on images scanned using any radiotracer, covering any area of the body, and representing any organism (human, mouse etc.).

USPET uses following packages: sys, scipy, numba, numpy, sklearn, cc3d, datetime. The main function call is

`uspet(image, clusters=100, extract_foreground="auto", do_cc=True, min_size=30, pca=5, location_weight=0.2, do_log=True, method="gmm")`

where the only mandatory argument image is a 4D array containing the original dynamic image. It can be loaded for example from a .nii file using package nibabel like this:

```python
import nibabel as nib

image = nib.load("path/to/my/pet/image/pet.nii").get_fdata()
segments = uspet(image)
```

and the obtained segmentation can be saved in a nifti format like this:

```python
nii_img = nib.Nifti1Image(segments, np.eye(4))
nib.save(nii_img, "path/to/my/segmented/pet/pet_segmented.nii")
```

or alternatively in numpy format (does not open nicely in most image viewer softwares, but convenient for further analyses in Python) like this:

```python
np.save("path/to/my/segmented/pet/pet_segmented.npy", segments)
```

### Arguments

The table below describes the parameters for USPET. The bolded default values are given in parenthesis.

| argument | explanation | 
| -------- | ----------- |
| image | 4D array, contains the actual dynamic image |
| clusters | integer (**100**), the number of segments to make (if do_cc is True, this will not be final) |
| extract_foreground | several options, indicates which voxels to use for segmentation, should be either <br>3D array (mask corresponding to the physical dimensions of the image,<br>2D array (foreground voxels as rows, x, y, and z coordinates as three cols),<br>**"auto"** (automatically detects the foreground voxels),<br>False (all voxels are used, don't use this, it is super slow).  |
| do_cc | boolean (**True**), defines if the output clusters should be broken into connected components |
| min_size | integer (**30**), defines how small segments are allowed |
| pca | integer (**5**) or None, the number of principal components |
| location_weight | float (**0.2**), a value between (inclusive) 0 and 1 indicating how much weight to put on location vs similarity of TACs. Value 0 means that location is not considered at all. |
| do_log | boolean (**True**), defines if log transformation should be made (usually a good idea) |
| method | string (**"gmm"**),  "gmm" or "k-means" defines the actual clustering method |

### Citation
If you use USPET in your research, please cite the original paper: FILL WHEN THE MANUSCRIPT IS ACCEPTED
