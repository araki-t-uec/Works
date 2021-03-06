#!/usr/bin/python3
# -*- coding:utf-8 -*-

import cv2, os, sys
import numpy as np
#import tensorflow as tf
#import keras
from keras.preprocessing.image import array_to_img, img_to_array
from keras import optimizers
from keras.models import Sequential, Model
from keras.utils import np_utils, plot_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
#from keras.layers.core import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from keras.applications.vgg16 import VGG16
from time import time 
video_dir = "./Img/Dogs/"
#classes = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
classes = ['Car', 'Drink', 'Feed', 'LookLeft', 'LookRight', 'Pet', 'PlayBall', 'Shake', 'Sniff', 'Walk', 'TurnRight']
class_num = len(classes)
train_x = []
train_y = []
epoch = 500
batch = 15 
savename = "dog_mean_vgg16finetune_33"
#savename="plt_test"
img_width, img_height = 320, 240


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
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        while(ret == True):
            frames.append(frame)
            ret, frame = cap.read()
            # cv2.imwrite('frame.png',gray)
        frames = (np.asarray(frames).mean(axis=0)) # the average overall frames
        #frames = (np.asarray(frames).std(axis=0)) # standard deviation frames
        train_x.append(frames)
        train_y.append(i)

        cap.release()

#cv2.destroyAllWindows()

#[0,1,2] -> [[1,0,0],[0,1,0],[0,0,1]]
train_y = np_utils.to_categorical(train_y, class_num)
train_x, train_y = np.asarray(train_x), np.asarray(train_y)
train_x = train_x.astype('float32')
train_x = train_x / 255 # 画素値を0から1の範囲に変換

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33)#, random_state=111) #rand seed



# VGG16のロード。FC層は不要なので include_top=False
input_tensor = Input(shape=(img_height, img_width, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層の作成
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(class_num, activation='softmax'))

# VGG16とFC層を結合してモデルを作成
vgg_model = Model(input=vgg16.input, output=top_model(vgg16.output))

# 最後のconv層の直前までの層をfreeze
for layer in vgg_model.layers[:15]:
    layer.trainable = False

# 多クラス分類を指定
vgg_model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
          metrics=['accuracy'])
# Fine-tuning
secc = time()
history = vgg_model.fit(train_x, train_y, batch_size=batch, epochs=epoch,
                    validation_data=(test_x, test_y), verbose=2)
cost = time()-secc

fig, (ac, los) = plt.subplots(ncols=2, figsize=(10,4))
ac.plot(history.history['acc'])
ac.plot(history.history['val_acc'])
ac.set_title('model accuracy')
ac.set_xlabel('epoch')
ac.set_ylabel('accuracy')
ac.legend(['acc', 'val_acc'], loc='lower right')
los.plot(history.history['loss'])
los.plot(history.history['val_loss'])
los.set_title('model loss')
los.set_xlabel('epoch')
los.set_ylabel('loss')
los.legend(['loss', 'val_loss'], loc='upper right')
#plt.savefig("Img/"+savename+"_acc.png")
plt.savefig("Img/"+savename+".png")
# 学習済みのModelを保存
open('./Models/'+savename+'.json',"w").write(vgg_model.to_json())
# 学習済みの重みを保存
vgg_model.save_weights('./Models/'+savename+'.h5')


# テストデータに適用
#predict_classes = vgg_model.predict_classes(test_x)
predict_classes = vgg_model.predict(test_x)

# マージ。yのデータは元に戻す
#mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(test_y, axis=1)})
mg_df = np.argmax(predict_classes,axis=1)

# confusion matrix
hge = pd.crosstab(mg_df['class'], mg_df['predict'])

print(hge)
print("fiting time: ", cost)
