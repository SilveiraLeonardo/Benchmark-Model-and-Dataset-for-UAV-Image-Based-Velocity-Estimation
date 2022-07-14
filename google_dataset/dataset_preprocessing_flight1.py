import os
import numpy as np
import matplotlib.pyplot as plt
import csv

# create CSV file and write header to it
header = ['file1_name', 'file2_name', "linear_velocity"]

with open('data_pairs_flight0.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

dataset_dir = os.listdir("100_0004")

print("[INFO] reading files...")

for i, file in enumerate(dataset_dir):

    if file[-3:] == "jpg" or file[-3:] == "JPG" :
        
        if (i == 0):
            old_image_file = file
        else:
            new_image_file = file
            distance_last_image = 0.0 # we dont have the ground truth
            
            # write to CSV data
            # ['image1_path', 'image2_path', "delta_position_meters"]
            text = [old_image_file, new_image_file, distance_last_image]
            with open('data_pairs_flight0.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(text)

            old_image_file = new_image_file


