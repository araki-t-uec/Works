#!/usr/bin/python3
# -*- coding:utf-8 -*-

import cv2, os, sys
import numpy as np
#import tensorflow as tf
#import keras
from keras.preprocessing.image import array_to_img, img_to_array
from keras.models import Sequential
from keras.utils import np_utils, plot_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from time import time
video_dir = "./Img/Dogs/"
#classes = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
classes = ['Car', 'Drink', 'Feed', 'LookLeft', 'LookRight', 'Pet', 'PlayBall', 'Shake', 'Sniff', 'Walk', 'TurnRight']
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
        train_x.append(frames)
        train_y.append(i)

        cap.release()

#cv2.destroyAllWindows()

#[0,1,2] -> [[1,0,0],[0,1,0],[0,0,1]]
train_y = np_utils.to_categorical(train_y, class_num)
# tmp = []
# for i in train_y:
#     l = [0]*class_num
#     l[int(i)-1] = 1
#     tmp.append(l)
# train_y = np.asarray(tmp)
train_x, train_y = np.asarray(train_x), np.asarray(train_y)
train_x = train_x.astype('float32')
train_x = train_x / 255 # 画素値を0から1の範囲に変換

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33)#, random_state=111) #rand seed

#モデルを作ってください．
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=train_x.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(class_num))       # クラスはn個
model.add(Activation('softmax'))

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])
plot_model(model, to_file="Img/Models/"+savename+"_model.png", show_shapes=True)

secc = time()
history = model.fit(train_x, train_y, batch_size=batch, epochs=epoch,
                    validation_data=(test_x, test_y), verbose=2)
cost = time()-secc
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(['acc', 'val_acc'], loc='lower right')
# plt.savefig("Img/"+savename+"_acc.png")

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
fig.savefig("Img/"+savename+".png")
# 学習済みのModelを保存
open('./Models/'+savename+'.json',"w").write(model.to_json())
# 学習済みの重みを保存
model.save_weights('./Models/'+savename+'.h5')


# テストデータに適用
predict_classes = model.predict_classes(test_x)

# マージ。yのデータは元に戻す
mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(test_y, axis=1)})

# confusion matrix
hge = pd.crosstab(mg_df['class'], mg_df['predict'])

print(hge)
print("fitting time: ", cost)
