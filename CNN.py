# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:40:00 2020

@author: Ravichandra Chintam
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


se = Sequential()

se.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation = "relu"))

se.add(MaxPooling2D(pool_size=(2,2)))

se.add(Flatten())

se.add(Dense(units = 128, activation = "relu"))

se.add(Dense(units = 1, activation = "sigmoid"))

se.compile(optimizer = "adam", loss = 'binary_crossentropy',metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('Thanos_Grimace/train_set',target_size = (64,64),batch_size = 10,class_mode = 'binary')

test_generator = test_datagen.flow_from_directory('Thanos_Grimace/test_set',target_size = (64,64),batch_size = 10,class_mode = 'binary')

se.fit_generator(train_generator,steps_per_epoch=412,epochs = 25,validation_data = test_generator,validation_steps = 45)



#Checking Extra Components
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Thanos_Grimace/_grimos/grimos_(5).jpg',target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
res = se.predict(test_image)
res = (res == 0)
train_generator.class_indices