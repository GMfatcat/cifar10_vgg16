# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:01:56 2020

@author: user
"""
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, ZeroPadding2D,Input,BatchNormalization, Dropout
from tensorflow.keras.models import Sequential,Model,load_model
import cv2
#from keras.models import Sequential,Model
from tensorflow.keras.optimizers import SGD, Adam 
from keras.utils import np_utils 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time
autoencoder2 = load_model('vgg123.h5')
print(autoencoder2.summary)