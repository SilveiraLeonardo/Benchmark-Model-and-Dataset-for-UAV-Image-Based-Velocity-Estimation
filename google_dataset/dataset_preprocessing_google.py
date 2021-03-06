import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

# create CSV file and write header to it
header = ['image1_path', 'image2_path', "delta_position_meters"]

with open('data_pairs_google_reta.csv', 'w', newline='') as f:
# with open('data_pairs_google.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

print("[INFO] reading files...")
mydf = pd.read_csv("dataset_google/details_route_reta.csv", sep=";")

for index, row in mydf.iterrows():

    if (int(index) == 0):
        old_image_file = row["file_name"]
    else:
        new_image_file = row["file_name"]
        distance_last_image = row["distance_last_image"]
        
        # write to CSV data
        # ['image1_path', 'image2_path', "delta_position_meters"]
        text = [old_image_file, new_image_file, distance_last_image]
        with open('data_pairs_google_reta.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(text)

        old_image_file = new_image_file
