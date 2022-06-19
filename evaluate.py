from model.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

MODEL_PATH = "checkpoints/model_22.hdf5"
TEST_HDF5 = "datasets/hdf5/test.hdf5"
VALID_HDF5 = "datasets/hdf5/val.hdf5"
BATCH_SIZE = 64

# load the pre-trained network
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

print("[INFO] loading generator...")
testGen = HDF5DatasetGenerator(TEST_HDF5, BATCH_SIZE)
valGen = HDF5DatasetGenerator(VALID_HDF5, BATCH_SIZE)

print("[INFO] predicting on the test data...")
predictions_test = model.predict_generator(testGen.generator(),
    steps=testGen.numImages//BATCH_SIZE, max_queue_size=10)
predictions_val = model.predict_generator(valGen.generator(),
    steps=valGen.numImages//BATCH_SIZE, max_queue_size=10)

mse = MeanSquaredError()
MSE_val = mse(valGen.db["labels"], predictions_val).numpy()
MSE_test = mse(testGen.db["labels"], predictions_test).numpy()

print("[INFO] valid - mean squared error: {}".format(MSE_val))
print("[INFO] test - mean squared error: {}".format(MSE_test))

testGen.close()