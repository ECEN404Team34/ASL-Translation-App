#Testing Code
from keras.models import load_model
import sklearn as sk
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

#loading the trained and validated model
current_model = load_model('January_gs_model.h5')


import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#gathering test data and setting parameters
test_datagen = ImageDataGenerator(
        rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    'grayscale_data/Test_report',
    batch_size = 1,
    class_mode = 'categorical',
    target_size = (64,64),
    shuffle=False
    )

#filling label arrays and converting via np
#true_lab = []
#for i in range(0,(len(test_gen))):
#    true_lab.extend(np.array(test_gen[i][1]))
#print("Evaluating on test data")


predictions = current_model.predict(test_gen)
predictions = np.argmax(predictions, axis=1)
#true_lab = np.argmax(true_lab, axis=1)

#creating confusion matrix and classification
#report
##cm = sk.metrics.confusion_matrix(true_lab, predictions)   #true_lab
#cr = sk.metrics.classification_report(true_lab, predictions)


a = 65 #ascii for character before A s.t. if image = A, pred = 1+64
#numpy array to letters
res = ""
predictions = [p + a for p in predictions]
for val in predictions:
    res = res + chr(val)

#print them
#print(cr)
#print(cm)
print(res)      #pred


#f = open('Metrics_Report.txt', 'w')
#f.write('Metrics for Image Set\n\nClassification Report\n\n{}\n\nConfusion Matrix\n'.format(cr, cm))
#f.close()

e = open('Predictions.txt', 'w')
e.write('Predictions for Given Images:\n\n{}'.format(res))
e.close()
