# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 21:12:38 2021

@author: mohap
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
# import tensorflow_hub as hub



main_path = "./archive/chest_xray/"


train_path = os.path.join(main_path,"train")
test_path = os.path.join(main_path,"test")
valid_path = os.path.join(main_path,"val")

batch_size = 16 

#The dimension of the images we are going to define is 500x500 
img_height = 500
img_width = 500
from tensorflow.keras.preprocessing.image import ImageDataGenerator# Create Image Data Generator for Train Set
image_gen = ImageDataGenerator(
                                  rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,          
                               )# Create Image Data Generator for Test/Validation Set
test_data_gen = ImageDataGenerator(rescale = 1./255)

train = image_gen.flow_from_directory(
      train_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary',
      batch_size=batch_size
      )
test = test_data_gen.flow_from_directory(
      test_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      shuffle=False,
      class_mode='binary',
      batch_size=batch_size
      )
valid = test_data_gen.flow_from_directory(
      valid_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary', 
      batch_size=batch_size
      )

dic = {0:'NORMAL', 1:'PNEUMONIA'}
plt.figure(figsize=(12, 12))
for i in range(0, 10):
    plt.subplot(2, 5, i+1)
    for X_batch, Y_batch in train:
        image = X_batch[0]        
        
        plt.title(dic.get(Y_batch[0]))
        plt.axis('off')
        plt.imshow(np.squeeze(image),cmap='gray',interpolation='nearest')
        break
plt.tight_layout()
plt.show()

cnn = models.Sequential([
                  layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)),
                  layers.MaxPooling2D(pool_size = (2, 2)),

                  layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)),
                  layers.MaxPooling2D(pool_size = (2, 2)),

                  layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)),
                  layers.MaxPooling2D(pool_size = (2, 2)),

                  layers.Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)),
                  layers.MaxPooling2D(pool_size = (2, 2)),

                  layers.Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)),
                  layers.MaxPooling2D(pool_size = (2, 2)),

                  layers.Flatten(),
                  layers.Dense(activation = 'relu', units = 128),
                  layers.Dense(activation = 'relu', units = 64),
                  layers.Dense(activation = 'sigmoid', units = 1)     
])



cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.summary()

early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss", patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
callbacks_list = [ early, learning_rate_reduction]

from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', np.unique(train.classes), train.classes)
cw = dict(zip( np.unique(train.classes), weights))
print(cw)

cnn.fit(train,epochs=25, validation_data=valid, class_weight=cw, callbacks=callbacks_list)

raw_cnn_test_accu = cnn.evaluate(test)
print('The testing accuracy is :',raw_cnn_test_accu[1]*100, '%')
