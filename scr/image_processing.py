import cv2
import glob
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

# Example image (used for testing/debugging)
image = cv2.imread('test_images/test3.jpg')


# ------------------------------
# Feature extraction functions
# ------------------------------

def bin_spatial(image, size=(32, 32)):
    """
    Spatial binning of color.
    Resize the image to a smaller size and flatten (ravel) into a 1D vector.
    Useful to capture coarse spatial color information.
    """
    return cv2.resize(image, size).ravel()


def color_hist(image, nbins, bins_range):
    """
    Compute color histogram features for each channel and concatenate them.
    
    Parameters:
        image      : input image (BGR or converted color space).
        nbins      : number of bins for histogram.
        bins_range : intensity range (e.g., (0,256)).
    
    Returns:
        hist_features : concatenated histogram of all 3 channels.
    """
    channel1_hist, _ = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist, _ = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist, _ = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)

    hist_features = np.concatenate((channel1_hist, channel2_hist, channel3_hist))
    return hist_features


def extract_color_features(image,
                           color_space='BGR',
                           size=(32, 32),
                           hist_nbins=32,
                           bins_range=(0, 256)):
    """
    Extract combined color features: spatial binning + color histograms.
    Optionally converts the image to another color space before extraction.
    """
    # Convert to desired color space
    if color_space == 'RGB':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    elif color_space == 'HSV':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   # (typo fixed: cv2.cvtColor, not cutColor)
    elif color_space == 'LUV':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    elif color_space == 'HLS':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    elif color_space == 'YUV':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    elif color_space == 'YCrCb':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        img = np.copy(image)

    # Spatial binning + histogram
    bin_features = bin_spatial(img, size)
    hist_features = color_hist(img, hist_nbins, bins_range)
    color_features = np.concatenate((bin_features, hist_features))

    return color_features


def extract_hog_features(image,
                         orient=9,
                         pix_per_cell=8,
                         cell_per_block=2,
                         vis=False,
                         feature_vec=True):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    
    Parameters:
        orient         : number of orientation bins.
        pix_per_cell   : pixels per cell.
        cell_per_block : cells per block.
        vis            : if True, also return a visualization image.
        feature_vec    : if True, return features as 1D array.
    
    Returns:
        hog_features : extracted HOG descriptor.
    """
    hog_features = hog(image,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       visualize=vis,
                       feature_vector=feature_vec)
    return hog_features


def extract_features(image_paths_list,
                     color_space='BGR',
                     size=(32, 32),
                     hist_nbins=32,
                     bins_range=(0, 256),
                     orient=9,
                     pix_per_cell=8,
                     cell_per_block=2,
                     vis=False,
                     feature_vec=True):
    """
    Extract features from a list of image file paths.
    Combines color features (spatial + histogram) and HOG features.
    """
    features = []
    for path in image_paths_list:
        image = cv2.imread(path)

        # Extract color-based features
        color_features = extract_color_features(image,
                                                color_space,
                                                size,
                                                hist_nbins,
                                                bins_range)

        # Convert to grayscale for HOG
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract HOG features
        hog_features = extract_hog_features(gray,
                                            orient,
                                            pix_per_cell,
                                            cell_per_block,
                                            vis,
                                            feature_vec)

        # Concatenate features
        features_vec = np.concatenate((color_features, hog_features))
        features.append(features_vec)

    return features


def extract_img_features(image,
                         color_space='BGR',
                         size=(32, 32),
                         hist_nbins=32,
                         bins_range=(0, 256),
                         orient=9,
                         pix_per_cell=8,
                         cell_per_block=2,
                         vis=False,
                         feature_vec=True):
    """
    Extract features from a single image (instead of list of paths).
    Useful when classifying patches/windows in sliding window search.
    """
    features = []

    # Color features
    color_features = extract_color_features(image,
                                            color_space,
                                            size,
                                            hist_nbins,
                                            bins_range)

    # HOG features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = extract_hog_features(gray,
                                        orient,
                                        pix_per_cell,
                                        cell_per_block,
                                        vis,
                                        feature_vec)

    # Combine
    features_vec = np.concatenate((color_features, hog_features))
    features.append(features_vec)
    return features


def create_scaler(x):
    """
    Create and fit a StandardScaler on feature data.
    This normalizes features to have zero mean and unit variance,
    which improves SVM performance.
    """
    return StandardScaler().fit(x)
