#!/usr/bin/python3
# -*- coding:utf-8 -*-

import cv2, os, sys
import numpy as np
#import tensorflow as tf
#import keras
# from keras.preprocessing.image import array_to_img, img_to_array
# from keras.models import Sequential
from keras.utils import np_utils, plot_model
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from sklearn.model_selection import train_test_split
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import pandas as pd
# from time import time
video_dir = "./Img/Dogs/"
#classes = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
classes = ['Car', 'Drink', 'Feed', 'LookLeft', 'LookRight', 'Pet', 'PlayBall', 'Shake', 'Sniff', 'Walk']
class_num = len(classes)
train_x = []
train_y = []
epoch = 200
batch = 30 
savename = "dog_mean"
#savename = "plt_test"

for i in range(class_num):
#    print(classes[i])
    datadir = os.path.join(video_dir,classes[i])
    #print(datadir)
    # = os.listdir(datadir)
    paths = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
    for j in paths:
        n = 0
        frames = []
        path = os.path.join(datadir,j)
#        print(path)
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        #while(cap.isOpened()):
        while(ret == True):
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #frame = img_to_array(frame)
            #frame = cv2.resize(frame, (24,24))
            frames.append(frame)
            ret, frame = cap.read()
            # cv2.imwrite('frame.png',gray)
        frames = (np.asarray(frames).mean(axis=0)) # the average overall frames
        print("./Img/"+j.split(".")[0]+".jpg")
        cv2.imwrite(("./Img/"+j.split(".")[0]+".jpg"),frames)

#        train_x.append(frames)
#        train_y.append(i)

        cap.release()

#cv2.destroyAllWindows()

# #[0,1,2] -> [[1,0,0],[0,1,0],[0,0,1]]
# train_y = np_utils.to_categorical(train_y, class_num)
# # tmp = []
# # for i in train_y:
# #     l = [0]*class_num
# #     l[int(i)-1] = 1
# #     tmp.append(l)
# # train_y = np.asarray(tmp)
# train_x, train_y = np.asarray(train_x), np.asarray(train_y)
# train_x = train_x.astype('float32')
# train_x = train_x / 255 # 画素値を0から1の範囲に変換

# np.save("dog_x.npy",train_x)
# np.save("dog_y.npy",train_y)

