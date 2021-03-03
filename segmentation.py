# Program that loads HandSeg data into TensorFlow

import numpy as np
import os
import PIL
# Imports
import PIL.Image
import tensorflow as tf
import pathlib
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Establish paths
home_path = '/home/ugrads/j/jaredhaley/ecen404'
data_path = '/mnt/lab_files/ECEN403-404/jaredhaley-21A/handseg-150k/'
data_path_images = '/mnt/lab_files/ECEN403-404/jaredhaley-21A/handseg-150k/images/'
data_path_masks = '/mnt/lab_files/ECEN403-404/jaredhaley-21A/handseg-150k/masks/'

# Change directory to data path
os.chdir(data_path)

# List contents of data path
files = os.listdir(data_path)

# TensorFlow version
print('TensorFlow version:',tf.__version__)

# Load in data to data_dir
data_dir = pathlib.Path(data_path)
data_dir_images = pathlib.Path(data_path_images)
data_dir_masks = pathlib.Path(data_path_masks)
# Find number of images in dataset
total = len(list(data_dir.glob('*/*.png')))
# Print image count
print('Total number of images in HandSeg:',total)

# Find number of images in /images/
images = list(data_dir.glob('images/*'))
image_count = len(images)
# Print image count
print('Images in HandSeg:',image_count)
# Print file name of first image in /images/
print(images[400])

# Find number of masks in /masks/
masks = list(data_dir.glob('masks/*'))
mask_count = len(masks)
# Print number of images in /masks/
print('Masks in HandSeg:',mask_count)
# Print file name of first image in /masks/
print(masks[400])

# File structure
# images
# 1.png, 2.png, 3.png...
# masks
# 1.png, 2.png, 3.png...

# Create dataset
# Batch is the number of training samples to work through before the model's internal parameters are updated
batch_size = 32
# img_height and img_width dictate the resolution of the photo
img_height = 200
img_width = 200

# This is incorrect because image_dataset_from_directory takes labels from folder names for a classifier.
# Create training dataset
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training",
                                                               #seed=123, image_size=(img_height, img_width),
                                                               #batch_size=batch_size)

# Create validation dataset
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,validation_split=0.2,subset="validation",seed=123,
                                                             #image_size=(img_height,img_width),batch_size=batch_size)

# I want the segmentation model to look at photo A from /images/, create a prediction, compare prediction
# to corresponding mask A in /masks/

# Display class names of training set
# class_names = train_ds.class_names
# print(class_names)

# Create the model

# Encoder

# Decoder

# Skip connections

# Train the model
