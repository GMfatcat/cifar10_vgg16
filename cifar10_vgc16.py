# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 11:04:13 2020

@author: user
"""


import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.utils 
from keras.utils import np_utils,plot_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, ZeroPadding2D,Input,BatchNormalization, Dropout,Dense
from tensorflow.keras.models import Sequential,Model,load_model
ishape = 32
VGG16_model = tf.keras.applications.VGG16(include_top=False,input_shape=(ishape,ishape, 3))

#%% data processing
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
x_train = X_train.astype('float32')/255
x_test = X_test.astype('float32')/255
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

print(x_train.shape)



#%%builld the model

from tensorflow.keras import models
new_model = Sequential()
new_model.add(VGG16_model)
new_model.add(Flatten())
new_model.add(Dense(512, activation='relu'))
new_model.add(Dropout(rate=0.25))
new_model.add(Dense(10, activation='softmax'))
print(new_model.summary())

#%% train the model

new_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

history = new_model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=20,
                    shuffle=True,
                    validation_split=0.2)
        

loss, accuracy = new_model.evaluate(x_test, y_test)
print('Test:')
print('Loss:', loss)
print('Accuracy:', accuracy)
new_model.save('vgg123.h5')

#將訓練過程損失可視化

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('vgg accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('vgg loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


autoencoder2 = load_model('vgg123.h5')
print(autoencoder2.summary())
#%% save npz


np.savez_compressed('train_data.npz',x_train = x_train,y_train=y_train,x_test=x_test,y_test=y_test)
print('finish')
















