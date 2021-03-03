# This program allows us to read files from the home directory to the data path directory

import os
import cv2 as cv

# Establish path
home_path = '/home/ugrads/j/jaredhaley/ecen404'
data_path = '/mnt/lab_files/ECEN403-404/jaredhaley-21A/handseg-150k/'
data_path_images = '/mnt/lab_files/ECEN403-404/jaredhaley-21A/handseg-150k/images/'
data_path_masks = '/mnt/lab_files/ECEN403-404/jaredhaley-21A/handseg-150k/masks/'

# Change directory to data path
os.chdir(data_path)

# List contents of data path
files = os.listdir(data_path_images)

# Iterate through files and verify that the program can read the file
x = 1
for file_name in files:
    while x <= 5:
        file_path = data_path_images + files[x]
        print(file_path)
        a = cv.imread(file_path)
        if a is None:
            print("Fail")
        else:
            print("Image read successfully")
        print(files[x]+'\n')
        x = x + 1
