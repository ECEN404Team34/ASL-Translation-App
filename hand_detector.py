import numpy as np
import os
import PIL
# Imports
import PIL.Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# TensorFlow version
print('TensorFlow version:',tf.__version__)

# Load in data to data_dir
data_dir = pathlib.Path('C:/Users/Jared/Desktop/ECEN 404/hello2/hands_nohands_data')
# Find image count in folder
image_count = len(list(data_dir.glob('*/*.jpg')))
print('Image total in detect_data:',image_count)

# Define photos with hands, display sample images
hands = list(data_dir.glob('hands/*'))
print(hands[0])
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

# Visualize the data - doesn't work because it's interactive? - it's intended to show a grid of images in the training set
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.title(class_names[labels[i]])
    plt.axis("off")
    plt.imshow(images[i].numpy().astype("uint8"))

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Create normalization layer [0,1] because [0,255] is too large
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

#AUTOTUNE = tf.data.AUTOTUNE

#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Two classes - [hands] and [nohands]
num_classes = 2

# Build the model
model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 1
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.show()

# Implement augmentation and data dropout
data_augmentation = tf.keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

# Create a new model with dropout
model = tf.keras.Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='DA Training Accuracy')
plt.plot(epochs_range, val_acc, label='DA Validation Accuracy')
plt.legend(loc='lower right')
plt.title('DA Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='DA Training Loss')
plt.plot(epochs_range, val_loss, label='DA Validation Loss')
plt.legend(loc='upper right')
plt.title('DA Training and Validation Loss')
#plt.show()

# Loading in new data for testing
x = 1
while x < 25:
    if x == 1:
        print("Not similar to training set")
    if x == 13:
        print("Similar to training set")
    xStr = str(x)
    test_path = pathlib.Path('C:/Users/Jared/Desktop/ECEN 404/hello2/detect_test/'+xStr+'.jpg')

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
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    x = x + 1

model.save('hand_detector_model.h5')
print("Model has been saved")














