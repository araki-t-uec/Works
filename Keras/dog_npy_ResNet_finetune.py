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
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from time import time
video_dir = "./Img/Dogs/"
#classes = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
classes = ['Car', 'Drink', 'Feed', 'LookLeft', 'LookRight', 'Pet', 'PlayBall', 'Shake', 'Sniff', 'Walk']
class_num = len(classes)
train_x = []
train_y = []
epoch = 300
batch = 15 
img_height,img_width=240,320
savename = "Resnet_finetune"
#savename = "plt_test"

train_x = np.load("dog_x.npy")
train_y = np.load("dog_y.npy")
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.20)#, random_state=111) #rand seed

#モデルを作ってください．
# VGG16のロード。FC層は不要なので include_top=False
input_tensor = Input(shape=(img_height, img_width, 3))
resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(img_height, img_width, 3), pooling=None)#, classes=1000)
# FC層の作成
top_model = Sequential()
top_model.add(Flatten(input_shape=resnet50.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(class_num, activation='softmax'))

# ResNet50とFC層を結合してモデルを作成
resnet_model = Model(input=resnet50.input, output=top_model(resnet50.output))

# 最後のconv層の直前までの層をfreeze
for layer in resnet_model.layers[:8]:
    layer.trainable = False

# 多クラス分類を指定
resnet_model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
          metrics=['accuracy'])

#plot_model(resnet.model, to_file="Img/Models/"+savename+"_model.png", show_shapes=True)

secc = time()
history = resnet_model.fit(train_x, train_y, batch_size=batch, epochs=epoch,
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
fig.savefig("Img/"+savename+".png")
# 学習済みのModelを保存
open('./Models/'+savename+'.json',"w").write(resnet_model.to_json())
# 学習済みの重みを保存
resnet_model.save_weights('./Models/'+savename+'.h5')


# テストデータに適用
predict_classes = resnet_model.predict(test_x)

# マージ。yのデータは元に戻す
mg_df = pd.DataFrame({'predict': np.argmax(predict_classes, axis=1), 'class': np.argmax(test_y, axis=1)})

# confusion matrix
hge = pd.crosstab(mg_df['class'], mg_df['predict'])

print(hge)
print("fitting time: ", cost)
