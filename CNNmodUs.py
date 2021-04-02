# Building the CNN

#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
import tensorflow as tf
import sklearn as sk
from sklearn.metrics import f1_score
import math
import gzip, os, re
from math import log

# Initialing the CNN
classifier = Sequential()

# Step 1 - Convolution Layer
classifier.add(Convolution2D(32, (3,  3), input_shape = (64, 64, 3), activation = 'relu'))

#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Adding second convolution layer
classifier.add(Convolution2D(32, (3,  3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Adding 3rd Convolution Layer
classifier.add(Convolution2D(64, (3,  3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Add 4th convolution layer and pooling
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(26, activation = 'softmax'))

#Compiling The CNN
classifier.compile(
              optimizer = optimizers.SGD(lr = 0.01),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])   #add f1 score

#Fitting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 35,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode = 'nearest')

validate_datagen = ImageDataGenerator(
        rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'ushands/Train',
        target_size=(64,64),       #64,64
        batch_size=32,
        class_mode='categorical')

validation_set = validate_datagen.flow_from_directory(
        'ushands/Validate',
        target_size=(64,64),       #64,64
        batch_size=32,
        class_mode='categorical')

model = classifier.fit_generator(
        training_set,
        steps_per_epoch=80,
        epochs=80,                                       #change back to 80
        validation_data = validation_set,
        validation_steps = (len(validation_set)/25)
      )


#'''#Saving the model
import h5py
#classifier.save('January_gs_model.h5')'''
classifier.save('Us_model.h5')








#models for tr/valid
print(model.history.keys())
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,2)
# summarize history for accuracy
axs[0,0].plot(model.history['accuracy'], label="Training Accuracy")
axs[0,0].plot(model.history['val_accuracy'], label="Validation Accuracy")
axs[0,0].set_title('Training and Validation Accuracy')
axs[0,0].legend(shadow=True, fancybox=True)

axs[0,1].plot(model.history['precision'], label="Training Precision")
axs[0,1].plot(model.history['val_precision'], label="Validation Precision")
axs[0,1].set_title('Training and Validation Precision')
axs[0,1].legend(shadow=True, fancybox=True)

axs[1,0].plot(model.history['recall'], label="Training Recall")
axs[1,0].plot(model.history['val_recall'], label="Validation Recall")
axs[1,0].set_title('Training and Validation Recall')
axs[1,0].legend(shadow=True, fancybox=True)

fig.subplots_adjust(hspace=0.5)
plt.show()



#Testing Code
from keras.models import load_model
import sklearn as sk
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

#loading the trained and validated model
current_model = load_model('Us_model.h5')


import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#gathering test data and setting parameters
test_datagen = ImageDataGenerator(
        rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    'ushands/Test',                              #change
    batch_size = 1,
    class_mode = 'categorical',
    target_size = (64,64),
    shuffle = False
    )

#filling label arrays and converting via np
true_lab = []
for i in range(0,(len(test_gen))):
    true_lab.extend(np.array(test_gen[i][1]))
print("Evaluating on test data")


predictions = current_model.predict(test_gen)
predictions = np.argmax(predictions, axis=1)
true_lab = np.argmax(true_lab, axis=1)

#creating confusion matrix and classification
#report
cm = sk.metrics.confusion_matrix(true_lab, predictions)   #true_lab
cr = sk.metrics.classification_report(true_lab, predictions)


a = 65 #ascii for character before A s.t. if image = A, pred = 1+64
#numpy array to letters
res = ""
predictions = [p + a for p in predictions]
for val in predictions:
    res = res + chr(val)

#print them
print(cr)
print(cm)
class LanguageModel(object):
  def __init__(self, word_file):
    # Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
    with gzip.open(word_file) as f:
      words = f.read().decode().split()
    self._wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
    self._maxword = max(len(x) for x in words)


  def split(self, s):
    """Uses dynamic programming to infer the location of spaces in a string without spaces."""
    l = [self._split(x) for x in _SPLIT_RE.split(s)]
    return [item for sublist in l for item in sublist]


  def _split(self, s):
    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
      candidates = enumerate(reversed(cost[max(0, i-self._maxword):i]))
      return min((c + self._wordcost.get(s[i-k-1:i].lower(), 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
      c,k = best_match(i)
      cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
      c,k = best_match(i)
      assert c == cost[i]
      # Apostrophe and digit handling (added by Genesys)
      newToken = True
      if not s[i-k:i] == "'": # ignore a lone apostrophe
        if len(out) > 0:
          # re-attach split 's and split digits
          if out[-1] == "'s" or (s[i-1].isdigit() and out[-1][0].isdigit()): # digit followed by digit
            out[-1] = s[i-k:i] + out[-1] # combine current token with previous token
            newToken = False
      # (End of Genesys addition)

      if newToken:
        out.append(s[i-k:i])

      i -= k

    return reversed(out)

DEFAULT_LANGUAGE_MODEL = LanguageModel(os.path.join(os.path.dirname(os.path.abspath(__file__)),'wordninja','wordninja_words.txt.gz'))
_SPLIT_RE = re.compile("[^a-zA-Z0-9']+")

def split(s):
  return DEFAULT_LANGUAGE_MODEL.split(s)   #Code inserted and altered from keredson on Github (https://github.com/keredson/wordninja/blob/master/wordninja/wordninja_words.txt.gz)
                                           #which was formed from Generic Human's answer here (https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words/11642687#11642687)
print(split(res))
