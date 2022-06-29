# import the necessary packages
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

mydf = pd.read_csv("reta_predito_esperado.csv")

predictions = mydf["predito"].to_numpy()
labels = mydf["esperado"].to_numpy()
error = mydf["erro"].to_numpy()

predictions = np.reshape(predictions, (-1, 1))
labels = np.reshape(labels, (-1, 1))
error = np.reshape(error, (-1, 1))

print("shape predictions: {}".format(predictions.shape))
print("shape labels: {}".format(labels.shape))
print("shape error: {}".format(error.shape))

accumulated_error = 0
accumulated_error_list = []
for i in range(100):
	accumulated_error = accumulated_error + error[i,0]
	accumulated_error_list.append(accumulated_error)

plt.style.use("ggplot")
plt.figure()
# plt.plot(np.arange(0, error.shape[0]), accumulated_error_list)
plt.plot(np.arange(0, 100), accumulated_error_list)
plt.xlabel("Frames")
plt.ylabel("Accumulated Error (m)")
# plt.legend()
plt.savefig("plots/accumulated_error_first100.png")

# plt.style.use("ggplot")
# plt.figure()
# plt.hist(error, bins=35)
# plt.title("Histogram of the Prediction Error")
# plt.ylabel("Frequency")
# plt.xlabel("Error (m)")
# # plt.legend()
# plt.savefig("plots/histogram_35.png")