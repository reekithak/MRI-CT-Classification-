# load json and create model
from keras.models import model_from_json
import cv2
import numpy as np
import pandas as pd
#classifier_json = classifier.to_json()
json_file = open(r'E:\DL_projects\VAC_multiModel\Model1-Classification\Procedure\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"E:\DL_projects\VAC_multiModel\Model1-Classification\Procedure\model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

img = cv2.imread(r'E:\DL_projects\VAC_multiModel\Model1-Classification\Train\Lung\MCUCXR_0002_0.png')
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])

classes = loaded_model.predict_classes(img)
if classes == 0:
    print("Brain")
else:
    print("Lung")
#print (classes)
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))