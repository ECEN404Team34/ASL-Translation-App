import os

import tensorflow as tf
from tensorflow import keras

new_model = tf.keras.models.load_model('hand_detector_model.h5')
print('Model loaded')
new_model.summary()

# verbose = 0 shows nothing, verbose = 1 with progress bar, verbose = 2 without progress bar
# loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
# print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

