#!/usr/bin/env python
# coding: utf-8

# In[77]:


import dlib as dlib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[79]:


classifier = Sequential()


# In[80]:


#conv1
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation= 'relu'))
#pool1
classifier.add(MaxPooling2D(pool_size=(2,2)))
#conv2
classifier.add(Convolution2D(32,3,3,activation='relu'))
#pool2
classifier.add(MaxPooling2D(pool_size=(2,2)))




# In[81]:


classifier.add(Flatten())


# In[82]:


#fc layer:

classifier.add(Dense(output_dim=128 , activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))


# In[83]:


#compiling

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[84]:


#Data augmentation

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator( rescale = 1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=15,
                                   vertical_flip=True,
                                   fill_mode='reflect',
                                   data_format='channels_last',
                                   brightness_range=[0.5, 1.5],
                                   featurewise_center=True,
                                   featurewise_std_normalization=True
                                  )
test_datagen = ImageDataGenerator(rescale = 1./255)


# In[85]:


#Flowing

training_set = train_datagen.flow_from_directory('C:/Users/samen/Train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('C:/Users/samen/Validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[86]:


#Training


hit = classifier.fit_generator(training_set,
                              samples_per_epoch = 128 ,
                               nb_epoch =50,
                               validation_data= test_set,
                               nb_val_samples = 59
                              )
#save to json
from keras.models import model_from_json
classifier_json = classifier.to_json()
with open(r"E:\DL_projects\VAC_multiModel\Model1-Classification\Procedure\classifier.json",'w') as json_file:
    json_file.write(classifier_json)
#saving weights hdf5
classifier.save_weights(r"E:\DL_projects\VAC_multiModel\Model1-Classification\Procedure\classifier.h5")


# In[89]:


fig = plt.figure()
plt.plot(hit.history['val_loss'])
plt.legend(['Validation'],loc='upper left')
plt.title('validation loss vs epoch')
plt.ylabel('validation loss')
plt.xlabel('Epoch')


# In[91]:



plt.plot(hit.history['accuracy'])
plt.legend(['validation'], loc='upper left')
plt.title('validation accuracy vs epoch')
plt.ylabel('validation accuracy')
plt.xlabel('Epoch')


# In[ ]:




