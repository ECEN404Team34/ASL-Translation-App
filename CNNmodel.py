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
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))         #128?
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dropout(0.5))
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
        'mydata/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

validation_set = validate_datagen.flow_from_directory(
        'mydata/valid_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

model = classifier.fit_generator(
        training_set,
        steps_per_epoch=800,
        epochs=50,                                       #change back to 50
        validation_data = validation_set,
        validation_steps = (len(validation_set)/50)
      )


'''#Saving the model
import h5py
classifier.save('Trained_model.h5')'''









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
current_model = load_model('Trained_model.h5')


import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#gathering test data and setting parameters
test_datagen = ImageDataGenerator(
        rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    'mydata/test_set',
    batch_size = 1,
    class_mode = 'categorical',
    target_size = (64,64)
    )

#filling label arrays and converting via np
true_lab = []
for i in range(0,(len(test_gen))):
    true_lab.extend(np.array(test_gen[i][1]))
print("Evaluating on test data")


predictions = current_model.predict(test_gen)       #batch_size=32
predictions = np.argmax(predictions, axis=1)
true_lab = np.argmax(true_lab, axis=1)

#creating confusion matrix and classification
#report
cm = sk.metrics.confusion_matrix(true_lab, predictions)   #true_lab
cr = sk.metrics.classification_report(true_lab, predictions)


#print them
print(cr)
print(cm)


f = open('Test Report.txt', 'w')
f.write('Metrics for Image Set\n\nClassification Report\n\n{}\n\nConfusion Matrix\n'.format(cr, cm))
f.close()
