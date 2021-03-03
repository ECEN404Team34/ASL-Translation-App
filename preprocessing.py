# This file takes a video as input and outputs a folder of segmented hand images

# Imports
import os
import cv2 as cv
import tensorflow as tf
import numpy as np
import pathlib
import shutil

# PHASE 1: INPUT A VIDEO, OUTPUT A FOLDER OF FRAMES
# Current problems: None

# "input" file folder waits for .mp4 to show up
dir_name = 'C:/Users/Jared/Desktop/ECEN 404/hello2/input_demo/'
print("Waiting for video file...")
while not os.listdir(dir_name):
    wait_variable = 0

print("Files have been found at " + dir_name)

# mp4 arrives via AWS (currently via drag and drop)

# Find file name
mypath = 'C:/Users/Jared/Desktop/ECEN 404/hello2/input_demo/'
onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

# Create file path based on video name assuming there's only one file at the specified directory
mypath = mypath + onlyfiles[0]

# Captures frame_test.mp4 and assigns it to cap
cap = cv.VideoCapture(mypath)

# Default is 60 fps
count = 1  # Number of total frames
freq = 1  # Save an image every "freq" frames
file = 1  # File numbering variable

# While the video is being processed
print("Video slicing has begun")
while cap.isOpened():
    fileStr = str(file)
    ret, frame = cap.read()
    if not ret:
        print("Video slicing has been completed")
        break
    if count % freq == 0:
        cv.imwrite('frames_demo/' + fileStr + '.jpg', frame)
        file = file + 1
    if cv.waitKey(1) == ord('q'):
        break
    count = count + 1

cap.release()
cv.destroyAllWindows()
print("Total frames:", count - 1)
print("Frames saved:", file - 1)

# PHASE 2: INPUT IMAGES FROM VIDEO, RECEIVE CONFIDENCE SCORES OF HAND VS NO HAND IN IMAGES
# Problems: In predictions section, find number of files and and create that as the limit for the while loop

# Run hands/nohands model

model = tf.keras.models.load_model('hand_detector_model.h5')
print('Model loaded')

# Load in data to data_dir
data_dir = pathlib.Path('C:/Users/Jared/Desktop/ECEN 404/hello2/hands_nohands_data')
# Find image count in folder
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Image total in detect_data:',image_count)

# Define photos with hands, display sample images
hands = list(data_dir.glob('hands/*'))
# print(hands[0])
# test1 = PIL.Image.open(str(hands[0]))
# test1.show()

# Define photos without hands, display sample images
nohands = list(data_dir.glob('nohands/*'))
# test2 = PIL.Image.open(str(nohands[0]))
# test2.show()

# Create dataset
batch_size = 10
img_height = 200
img_width = 200

train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split = 0.2, subset="training",
                                                               seed=123, image_size = (img_height, img_width),
                                                               batch_size = batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,validation_split=0.2,subset="validation",seed=123,
                                                             image_size=(img_height,img_width),batch_size = batch_size)
class_names = train_ds.class_names
print(class_names)

# If confidence score of hand > 85%
# Pass images to has_hand folder
has_hand_path = 'C:/Users/Jared/Desktop/ECEN 404/hello2/has_hand'
# Loading in data for testing
x = 1
while x < 25:
    if x == 1:
        print("Not similar to training set")
    if x == 13:
        print("Similar to training set")
    xStr = str(x)
    test_path = 'C:/Users/Jared/Desktop/ECEN 404/hello2/frames_demo/'+xStr+'.jpg'

    img = tf.keras.preprocessing.image.load_img(
        test_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print('Image:',x)
    print(
        "Image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score)))
    print(class_names[np.argmax(score)])
    if class_names[np.argmax(score)] == 'hands' and 100*np.max(score) >= 85:
        shutil.copy2(test_path, has_hand_path)
        print("moved")
    x = x + 1

# PHASE 3: INPUT IMAGES, RECEIVE LOCALIZED IMAGES CONFIDENCE SCORES OF HAND LOCALIZATION IN IMAGES
# Status: Incomplete
# Problems:

# Run hand detection model

# If hand detection model > 85%
# Pass images to hand_detected folder

# PHASE 4: INPUT IMAGES, RECEIVE

# END OF SUBSYSTEM - SEND IMAGES TO CLASSIFIER
