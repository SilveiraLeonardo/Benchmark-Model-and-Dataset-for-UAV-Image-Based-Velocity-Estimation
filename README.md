# Benchmark-Model-and-Dataset-for-UAV-Image-Based-Velocity-Estimation

Authors: Leonardo Silveira - Instituto Tecnológico de Aeronáutica / 
Mateus Rodrigues de Barros - Instituto Tecnológico de Aeronáutica

We introduced a large-scale dataset for UAV linear velocity estimation. We used AirSim to record in-flight images of drones flying in different scenarios and with different atmospheric conditions. Additionally, we used this dataset to train a deep learning model for the vehicle velocity estimation. This trained network can be used as an initial baseline for other researchers that may use the dataset, as well as a starting model for real world applications of drone speed estimation.

Links para o dataset:

Dataset - part1:
https://zenodo.org/record/6670189#.YrDt3HbMLIU

Dataset - part2:
https://zenodo.org/record/6671685#.YrDt3XbMLIU

Dataset - part3:
https://zenodo.org/record/6672408#.YrDt3nbMLIU

Python scripts:

For preparing the data:

1) data_preprocessing.py: creates a CSV file with all the image pairs and note pairs file paths, and their labels (velocity of the drone in the second image)
2) data_preprocessing_2.py: creates a CSV file with all the image pairs and note pairs file paths, and their labels (position delta between the two images)
3) build_data_splits.py: consumes the CSV file created by the data_preprocessing.py scripts, and creates 3 CSVs - train.csv, val.csv and test.csv. Each file contains the path to the pair of images and the label for the regression problem. Additionally, a Json file with the mean values of the red, green and blue channels of the images is also saved.
4) build_hdf5_datasets.py: consumes the CSV files train.csv, val.csv and test.csv, and create 3 .hdf5 databases (training, testing and validation), each containing: pair of images, resized to 128x128 and grayscaled, and their labels ([image1], [image2], [label]).
