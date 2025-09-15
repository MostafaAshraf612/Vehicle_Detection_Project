import glob
import numpy as np
from image_processing import extract_features   # custom function: extracts color + HOG features
from image_processing import create_scaler      # custom function: creates StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def create_train_test_data(cars_features, non_cars_features):
    """
    Combine car and non-car features, create labels, split into train/test,
    and scale features.
    
    Parameters:
        cars_features      : list/array of features extracted from car images.
        non_cars_features  : list/array of features extracted from non-car images.
    
    Returns:
        x_train, x_test, y_train, y_test : training/testing data and labels
        x_scaler                        : fitted StandardScaler (to use later on test data)
    """

    # Stack car + non-car features into one array
    x = np.vstack((cars_features, non_cars_features)).astype(np.float64)  
    # Create labels: 1 for cars, 0 for non-cars
    y = np.hstack((np.ones(len(cars_features)), np.zeros(len(non_cars_features))))

    # Split dataset: 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Fit a StandardScaler on training data
    x_scaler = create_scaler(x_train)

    # Normalize train & test features
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, x_scaler


# -------------------------------------------------
# 1. Collect training image paths
# -------------------------------------------------
# Recursively grab all PNG images in 'vehicles' and 'non-vehicles' folders
cars_images_paths = glob.glob('vehicles/**/*.png')
non_cars_images_paths = glob.glob('non-vehicles/**/*.png')

# Extract features for car and non-car datasets
cars_features = extract_features(cars_images_paths)
non_cars_features = extract_features(non_cars_images_paths)


# -------------------------------------------------
# 2. Train SVM Classifier
# -------------------------------------------------
svc = LinearSVC()   # Linear Support Vector Classifier

# Prepare train/test split + scaling
x_train, x_test, y_train, y_test, x_scaler = create_train_test_data(cars_features, non_cars_features)

# Train classifier
svc.fit(x_train, y_train)

# Evaluate accuracy on test set
acc = svc.score(x_test, y_test)
print('✅ Accuracy on test set: {:.4f}'.format(acc))
