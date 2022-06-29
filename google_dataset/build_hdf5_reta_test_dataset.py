from model.hdf5datasetwriter import HDF5DatasetWriter
from model.aspectawarepreprocessor import AspectAwarePreprocessor
import cv2
import imutils
import pandas as pd
import numpy as np
import progressbar

def preprocess(image):
    # resize to 128x128x3
    # to grayscale
    aap = AspectAwarePreprocessor(128, 128)
    image = aap.preprocess(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=-1)
    return image

# TRAIN_HDF5 = "datasets/hdf5/train.hdf5"
# VALID_HDF5 = "datasets/hdf5/val.hdf5"
# TEST_HDF5 = "datasets/hdf5/test.hdf5"

# TRAIN_HDF5 = "datasets/hdf5/train_position.hdf5"
# VALID_HDF5 = "datasets/hdf5/val_position.hdf5"
# TEST_HDF5 = "datasets/hdf5/test_position.hdf5"

# TRAIN_HDF5 = "dataset_google/hdf5/train_google.hdf5"
# VALID_HDF5 = "dataset_google/hdf5/val_google.hdf5"
# TEST_HDF5 = "dataset_google/hdf5/test_google.hdf5"

TEST_HDF5 = "dataset_google/hdf5/test_reta.hdf5"

# # train = pd.read_csv("lists/train.csv")
# # train = pd.read_csv("lists/train_position.csv")
# train = pd.read_csv("lists/train_google.csv")
# train_path1 = train["path1"]
# train_path2 = train["path2"]
# train_labels = train["label"]

# # validation = pd.read_csv("lists/val.csv")
# # validation = pd.read_csv("lists/val_position.csv")
# validation = pd.read_csv("lists/val_google.csv")
# validation_path1 = validation["path1"]
# validation_path2 = validation["path2"]
# validation_labels = validation["label"]

# test = pd.read_csv("lists/test.csv")
# test = pd.read_csv("lists/test_position.csv")
test = pd.read_csv("lists/data_pairs_google_reta.csv")
test_path1 = test["image1_path"]
test_path2 = test["image2_path"]
test_labels = test["delta_position_meters"]

datasets = [(test_path1, test_path2, test_labels, TEST_HDF5)]

for (paths1, paths2, labels, outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths1), 128, 128, 1), outputPath)

    # initialize progressbar
    widgets = ["Building dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths1), widgets=widgets).start()

    for (i, (path1, path2, label)) in enumerate(zip(paths1, paths2, labels)):

        # adjust the path if it is the google dataset
        path1 = "dataset_google/reta/" + path1
        path2 = "dataset_google/reta/" + path2

        image1 = cv2.imread(path1)
        image1 = preprocess(image1)

        image2 = cv2.imread(path2)
        image2 = preprocess(image2)    

        writer.add([image1], [image2], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()
