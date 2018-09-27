#!/usr/bin/python3
# -*- coding:utf-8 -*-
import sys, os
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras import optimizers
import cv2
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import pandas as pd


datadir="Img/UECFOOD100_named/"
dirs = os.listdir(datadir)
classes = [f for f in dirs if os.path.isdir(os.path.join(datadir, f))]

nb_classes = len(classes)
size = (150, 150)
train_x = []
train_y = []

for i in range(0,len(classes)):
    #print("load images from " + str(i) + ": "+classes[i])
    path = datadir+classes[i]
    files = os.listdir(path)
    for f in files:
        # print(f,i)
        try:
            img_path = os.path.join(path,f)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size)
            #img = img_to_array(load_img(img_path))#, target_size=(64,64))
            train_x.append(img)
            train_y.append(i) #tag number
        except:
            #print(f+" is not img file..")
            pass

train_x = np.asarray(train_x) # list --to--> numpy.ndarray
train_x = train_x.astype('float32')
train_x = train_x / 255 # 画素値を0から1の範囲に変換

tmp = []

for i in train_y:
    l = [0]*len(classes)
    l[i] = 1
    tmp.append(l)
train_y = np.asarray(tmp)
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33)#, random_state=111)
input_tensor = Input(shape=(150,150, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層の作成
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))

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
history = vgg_model.fit(train_x, train_y, batch_size=8, epochs=200,
                        validation_data=(test_x, test_y), verbose=1)
# history = vgg_model.fit_generator(
#     train_generator,
#     samples_per_epoch=nb_train_samples,
#     nb_epoch=nb_epoch,
#     validation_data=validation_generator,
#     nb_val_samples=nb_validation_samples)

# 重みを保存
vgg_model.save_weights(os.path.join("./Models/", 'vgg16_finetune_to_food.h5'))

# out put accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.savefig("Img/vgg16_uecfood100_finetune_acc.png")

# plt.plot(range(1, epochs+1), result.history['acc'], label="training")
# plt.plot(range(1, epochs+1), result.history['val_acc'], label="validation")
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig("Img/finetune_vgg16model.png")
