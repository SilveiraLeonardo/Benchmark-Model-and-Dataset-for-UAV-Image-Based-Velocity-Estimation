from model.hdf5datasetgenerator import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import csv

# Script correct for evaluation of the model in the validation and testing datasets

MODEL_PATH = "checkpoints_google/model_checkpoint"
# MODEL_PATH = "checkpoints_google/fine_tuned_model_checkpoint"

TEST_HDF5 = "datasets/hdf5/flight0.hdf5"
BATCH_SIZE = 1

# create CSV file and write header to it
header = ['deslocamento_predito', 'deslocamento_predito_acumulado']

with open('reta_predito_flight0.csv', 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerow(header)

# load the pre-trained network
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

print("[INFO] loading testing generator...")
testGen = HDF5DatasetGenerator(TEST_HDF5, BATCH_SIZE)

gen_test = testGen.generator()

trajectory_length_predicted = 0
n_pairs = 0
for i in range(testGen.numImages // BATCH_SIZE):

	image_pair, _ = next(gen_test)
	
	predictions = model.predict(image_pair)

	trajectory_length_predicted = trajectory_length_predicted + float(predictions[0])

	n_pairs = n_pairs + image_pair[0].shape[0]

	text = [float(predictions[0]), float(trajectory_length_predicted)]
	with open('reta_predito_flight0.csv', 'a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(text)

print("[INFO] accumulated trajectory predicted of {} m in {} pairs of images...".format(trajectory_length_predicted, n_pairs))
testGen.close()


