import sys
from scipy.ndimage import gaussian_filter
from numba import jit, prange
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import cc3d
from scipy.ndimage import grey_dilation
from scipy import stats
from datetime import datetime


def foreground_auto(image):

    mean_per_voxel = np.mean(image, axis=3)
    mask = 1*(mean_per_voxel > np.mean(mean_per_voxel)) + 1
    connected_components = cc3d.connected_components(mask, connectivity=6)
    labels, counts = np.unique(connected_components, return_counts=True)

    # Detect the main cc's for background and foreground
    index_background, index_foreground = -1, -1
    decreasing_index, ind = np.argsort(-counts), 0
    while (index_background < 0) | (index_foreground < 0):
        current_index = decreasing_index[ind]
        current_size, current_label = counts[current_index], labels[current_index]
        current_type = mask[np.where(connected_components == current_label)][0]
        if (current_type < 1.5) & (index_background < 0):
            index_background = current_index
        if (current_type > 1.5) & (index_foreground < 0):
            index_foreground = current_index
        ind = ind + 1

    # Get rid of all smaller cc's by melting them to their surrounding
    melt_labels = np.delete(labels, np.array([index_foreground, index_background]))
    mask[np.isin(connected_components, melt_labels)] = mask[np.isin(connected_components, melt_labels)] + 1
    mask[mask > 2.5] = 1

    # Construct and return foreground indices as a 2D array (cols: coordinates, rows: voxels)
    indices = np.transpose(np.array(np.where(mask > 1.5)))
    return indices


# Input: 'clusters' is a 3D array including clustering results after connected component analysis (0 -> background),
#        'min_size' is a minimum number of voxels accepted as a segment
# Output: Melts the small segments (cutoff: min_size) into the surrounding segment closest to it. Can keep the small
#         segments if their mean tac differ enough from all the neighbour segments'. Assumes connected components.
def melt_to_background(clusters, min_size):

    # Clean off the small segments with size below the cutoff (unless they are very different from surrounding segments)
    labels, counts = np.unique(clusters, return_counts=True)
    melt_indices = np.where(counts < min_size)[0]

    # Find and merge the smallest segment
    for i in range(len(melt_indices)):

        # Identify all potential background segments neighbouring the small segment
        small_ind = melt_indices[i]
        small_label = labels[small_ind]
        lab_mask = np.ones(clusters.shape) * (clusters == small_label)
        mask_extended = grey_dilation(lab_mask, size=(3, 3, 3))
        neighbour_labels, neighbour_counts = np.unique(clusters[mask_extended > 0.5], return_counts=True)

        # Identify the label with most neighbouring instances and merge small segment into it
        closest_neighbour = neighbour_labels[np.argmax(neighbour_counts)]
        clusters[np.where(lab_mask > 0.5)] = closest_neighbour

    return clusters


# Input: 'flat_tacs' is a 2D array containing the (preprocessed) voxel values (voxels as rows, features as cols),
#        'dims' is a tuple including the original image dimensions
#        'indices' is a 2D array including the coordinates (cols) of the foreground voxels (rows),
#        'cluster_number' is the number of clusters to use,
#        'method' indicate how to do the clustering, should be "gmm" or "k-means"
# Output: Returns a 3D array including the image's cluster labels
def cluster_tacs(flat_tacs, dims, indices, cluster_number, method):

    if method == "gmm":
        attempt = 0
        while attempt < 10:
            try:
                gmm_model = GaussianMixture(n_components=cluster_number).fit(flat_tacs)
                cluster_labels = gmm_model.predict(flat_tacs) + 1
                attempt = 15
            except:
                attempt = attempt + 1

        # Return nan if GMM didn't succeed in 10 tries
        if attempt == 10:
            cluster_labels = np.nan

    if method == "k-means":
        cluster_labels = KMeans(n_clusters=cluster_number).fit_predict(flat_tacs)

    image_clusters = np.zeros(tuple((dims[0], dims[1], dims[2])))
    for ind in range(indices.shape[0]):
        image_clusters[indices[ind, 0], indices[ind, 1], indices[ind, 2]] = cluster_labels[ind] + 1

    return image_clusters
    

# Helper function for clean_labels    
@jit(parallel=True, nopython=True)
def label_means(clusters_3d, labels, original):

    clusters_flat = clusters_3d.flatten()
    original_flat = original.flatten()
    label_mean = np.empty(len(labels))
    for lab in prange(len(labels)):
        index = np.where(clusters_flat == labels[lab])[0]
        values = original_flat[index]
        label_mean[lab] = np.mean(values)

    return label_mean    


# Input: 'segments' is a 3D array including the segment labels,
#        'image' is the original 4D array of the dynamic image
# Output: Returns a 3D array similar to the input 'segments', but different segments are re-labeled so that 
#         the asegment with the highest mean intensity in the original image gets the highest integer label 
@jit(nopython=True, parallel=True)
def clean_labels(segments, image):

    segments_flat = segments.flatten()
    segment_labels = np.unique(segments_flat)
    image_activity = np.sum(image, axis=3)
    segment_activity = label_means(segments, segment_labels, image_activity)
    activity_rank = np.argsort(segment_activity)
    final_segments = np.zeros(np.prod(np.asarray(segments.shape)))
    for new_label in prange(len(segment_activity)):
        old_label = segment_labels[activity_rank[new_label]]
        final_segments[np.where(segments_flat == old_label)[0]] = new_label

    return np.reshape(final_segments, segments.shape)


# Input: 'image' is a 4D array containing the actual image
#        'clusters' is the number of segments to make (if do_cc is True, this will not be final)
#        'extract_foreground' indicates which voxels to use for segmentation, should be either 
#                             3D array (mask corresponding to the physical dimensions of the image, 
#                             2D array (foreground voxels as rows, x, y, and z coordinates as cols), 
#                             "auto" (automatically assumes voxels with ), or
#                             False (all voxels are used, don't use this, it is super slow). 
#        'do_cc' is a boolean defining if the output clusters should be broken into connected components,
#        'min_size' is an integer defining how small segments are allowed,
#        'pca' is the number of principal components (or None if scaled TACs used directly)
#        'location_weight' is a vlaue between (inclusive) 0 and 1 indicating how much weight to put on 
#                          location vs similarity of TACs. Value 0 means that location is not considered at all.
#        'do_log' is a blloean and defines if log transformation should be made (usually a good idea)
#        'method' defines the actual clustering method. Either "gmm" or "k-means"
# Output: Returns a 3D array corresponding to the first three physical dimensions of the input 'image'. The array
#         is filled with integers representing different segments.  
def uspet(image, clusters=100, extract_foreground="auto", do_cc=True, min_size=30, pca=5, location_weight=0.2,
          do_log=True, method="gmm"):

    # If asked for, identify which voxels to use as foreground
    now = datetime.now()
    print(now.strftime("%H:%M:%S"), "Extracting foreground from the background")
    if isinstance(extract_foreground, bool):
        if not extract_foreground:
            indices = np.transpose(np.array(np.where(image[:, :, :, 0] > -1)))  # Can this be a problem in some cases?
        else:  # Same than 'auto'
            indices = foreground_auto(image)
    elif isinstance(extract_foreground, str):
        if extract_foreground == "auto":
            indices = foreground_auto(image)
        else:
            sys.exit('Argument "extract_foreground" should be one of these: 3D array, 2D array, "auto", False, True (-> "auto").')
    elif isinstance(extract_foreground, np.ndarray):
        # Assumes that the argument is a mask
        if len(extract_foreground.shape) == (len(image.shape) - 1):
            indices = np.transpose(np.array(np.where(extract_foreground > 0.5)))
        # Foreground array can't have more dimensions than the static version of the analysed image
        elif len(extract_foreground.shape) >= len(image.shape):
            sys.exit('Argument "extract_foreground" can not have higher dimension than the physical dimensions of the '
                     'input image.')
        # A 2D array is expected to be the foreground index array directly
        else:
            indices = extract_foreground
    else:
        sys.exit('Argument "extract_foreground" should be one of these: 3D array, 2D array, "auto", False, True.')

    # Denoise image
    now = datetime.now()
    print(now.strftime("%H:%M:%S"), "Preprocessing the image")
    image = gaussian_filter(image, sigma=1)

    # Do log2 transformation if asked for. NOTE: +1 is potentially troublesome with SUVs as it is proportionally big
    if do_log:
        for i in range(image.shape[3]):  # Loop to avoid memory errors
            image[:, :, :, i] = np.log2(image[:, :, :, i] + 1)

    # Add location info and scale the image
    flat_tacs = np.array([image[i[0], i[1], i[2], :] for i in indices])
    if location_weight is not None:
        flat_tacs = np.column_stack((flat_tacs, indices))
        flat_tacs_scaled = stats.zscore(flat_tacs, axis=0)
        time_points = image.shape[3]
        weight = time_points * location_weight / (1 - location_weight) / 3
        flat_tacs_scaled[:, time_points:(time_points + 3)] *= weight  # Multiplies the given part with weight
    else:
        flat_tacs_scaled = stats.zscore(flat_tacs, axis=0)
    flat_tacs_scaled[np.where(np.isnan(flat_tacs_scaled))] = 0

    # Dimensionality reduction
    if pca is not None:
        flat_tacs_pca = PCA(n_components=pca).fit_transform(flat_tacs_scaled)
    else:
        flat_tacs_pca = flat_tacs_scaled

    # Do the segmentation
    now = datetime.now()
    print(now.strftime("%H:%M:%S"), "Clustering")
    dims = (image.shape[0], image.shape[1], image.shape[2])
    image_clusters = cluster_tacs(flat_tacs_pca, dims, indices, clusters, method)

    # Connected component analysis
    if do_cc:
        now = datetime.now()
        print(now.strftime("%H:%M:%S"), "Cleaning small segments after connected component analysis")
        connected_components = cc3d.connected_components(image_clusters, connectivity=26)
        image_clusters = melt_to_background(clusters=connected_components, min_size=min_size)

    # Clean labels so that the segment with the highest activity (sum over time) gets the highest label
    now = datetime.now()
    print(now.strftime("%H:%M:%S"), "Cleaning the labels")
    image_clusters = clean_labels(image_clusters, image)
    print(now.strftime("%H:%M:%S"), "Done")
    return image_clusters

