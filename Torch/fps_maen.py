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
video_dir = "./data/Dogs/Resque/Cv2_Optical_Movie/"
#classes = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
classes = ['bark','cling','command','eat-drink','look_at_handler','run','see_victim','shake','sniff','stop','walk-trot']
class_num = len(classes)
train_x = []
train_y = []
epoch = 200
batch = 20 
savename = "dog_mean_opt"
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
        #filepath = "./data/Dog_opt_mean/"+classes[i]+"/"+j.split(".")[0]+".jpg"
        filepath = "./data/Dog_opt_mean/"+str(i+1)+"/"+j.split(".")[0]+".jpg"
        print(filepath, classes[i])
        #print("./data/Dog_mean/"+j.split(".")[0]+".jpg")
        cv2.imwrite(filepath,frames)
        #cv2.imwrite(("./data/Dog_mean/"+j.split(".")[0]+".jpg"),frames)

#        train_x.append(frames)
#        train_y.append(i)

        cap.release()
