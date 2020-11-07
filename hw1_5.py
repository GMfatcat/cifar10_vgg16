# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\user\Desktop\hw1_5.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, UpSampling2D, ZeroPadding2D,Input,BatchNormalization, Dropout
from tensorflow.keras.models import Sequential,Model,load_model
import cv2
from tensorflow.keras.optimizers import SGD, Adam 
from keras.utils import np_utils 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time
import sys
import  random
from datetime import datetime
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(264, 486)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 80, 111, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(70, 150, 111, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(70, 220, 111, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(70, 280, 111, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(70, 370, 81, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setGeometry(QtCore.QRect(70, 340, 81, 22))
        self.spinBox.setObjectName("spinBox")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 264, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        #load image
        self.pushButton.clicked.connect(self.button1_click)
        #show parameter
        self.pushButton_2.clicked.connect(self.button2_click)
        #show model structure
        self.pushButton_3.clicked.connect(self.button3_click)
        #show model process
        self.pushButton_4.clicked.connect(self.button4_click)
        # test
        self.pushButton_5.clicked.connect(self.button5_click)
        #spin box value
        self.spinBox.valueChanged.connect(self.Valuechange)
    
    def Valuechange(self):
        global select_number
        select_number = self.spinBox.value()
        
    
    def button1_click(self):
        global x_test,y_test
        print('loading npz file')
        file = np.load('train_data.npz')
        x_train = file['x_train']
        y_train = file['y_train']
        x_test = file['x_test']
        y_test = file['y_test']
        print('----finish-----')
        for _ in range(10):
            random.seed(datetime.now())
            num = random.randint(0,50000)
            cv2.namedWindow('this is No.' + str(num) + ' image', cv2.WINDOW_NORMAL)#讓window可自由調整大小 不然看不到
            cv2.imshow('this is No.' + str(num) + ' image',x_train[num])
            print(y_train[num])
            label = np.argmax(y_train[num]) + 1 # 讓數字從1開始
            list = {1:'airplane' , 2:'automobile', 3:'bird' , 4:'cat' , 5:'deer' , 6:'dog' , 7:'frog' , 8:'horse' , 9: 'ship' , 10:'truck'}
            ans = list.get(label)
            print('This is a',ans)
            cv2.waitKey(0)
        print('finish showing 10 random images\n')

    def button2_click(self):
        print('batch size: 20\nlearning rate : 0.01 \noptimizer : adam \nepoch : 50 \nloss= categorical_crossentropy')
        print('----finish-----')
    def button3_click(self):
        global vgg
        vgg = load_model('vgg123.h5')
        print('loading model')
        print(vgg.summary())
        print('---finish---')
    def button4_click(self):
        loss = cv2.imread(r'loss.png',1)
        accuracy = cv2.imread(r'accuracy.png',1)
        img = np.hstack((loss,accuracy))
        cv2.imshow('loss & accuracy',img)
        cv2.waitKey(0)
    def button5_click(self):
        global vgg,x_test,y_test,select_number
        test = np.empty((1,32,32,3))
        test[0] = x_test[select_number]
        result = vgg.predict(test)
        print(result)
        x = ['airplane' , 'automobile', 'bird' , 'cat' , 'deer' , 'dog' , 'frog' ,'horse' , 'ship', 'truck']
        y = list(result.transpose())
        plt.title("probability graph") 
        plt.ylabel('probability')
        plt.xlabel('10 categories')
        plt.plot(x, y, color ="red")
        plt.show()
        cv2.namedWindow('image you select', cv2.WINDOW_NORMAL)#讓window可自由調整大小 不然看不到
        cv2.imshow('image you select',x_test[select_number])
        label = np.argmax(result) + 1 # 讓數字從1開始
        list1 = {1:'airplane' , 2:'automobile', 3:'bird' , 4:'cat' , 5:'deer' , 6:'dog' , 7:'frog' , 8:'horse' , 9: 'ship' , 10:'truck'}
        ans = list1.get(label)
        print('This is a',ans)
        cv2.waitKey(0)
        print('----finish-----')
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "show train image"))
        self.pushButton_2.setText(_translate("MainWindow", "show hyperparameters"))
        self.pushButton_3.setText(_translate("MainWindow", "show model structure"))
        self.pushButton_4.setText(_translate("MainWindow", "show training process"))
        self.pushButton_5.setText(_translate("MainWindow", "test image"))

if __name__ == '__main__':  
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()

    ui.setupUi(MainWindow) 
    MainWindow.show()
    sys.exit(app.exec_())