#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import os, sys
#import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, plot_model
from keras.optimizers import SGD
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd





datadir="Img/Make3class/"
dirs = os.listdir(datadir)
addition = [f for f in dirs if os.path.isdir(os.path.join(datadir, f))]
#addition=["top","middle","bottom"] #,"test"]
datas = {}
train_x = [] 
train_y = []
for i in range(0,len(addition)):
    path = datadir+addition[i]
    files = os.listdir(path)
    datas[addition[i]]=([f for f in files if os.path.isfile(os.path.join(path, f))])
    if i < 3 :
        for j in datas[addition[i]]:
            img = img_to_array(load_img(os.path.join(path,j)))#, target_size=(64,64)))
            train_x.append(img)
            train_y.append(i) #tag number(top,middle,bottom )


train_x = np.asarray(train_x) # list --to--> numpy.ndarray
train_y = np.asarray(train_y)

train_x = train_x.astype('float32')
train_x = train_x / 255 # 画素値を0から1の範囲に変換

print(train_y)
exit()


k = 3 #(top, middle, bottom)
train_y = np_utils.to_categorical(train_y, k) #[0,1,2] -> [[1,0,0],[0,1,0],[0,0,1]]
# 学習用データとテストデータ
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33)#, random_state=111) #rand seed

epoch = 20
opstep = 5
epoch0 = 1
batch = 150  

lr = 0.1

#temp_img = load_img(datadir+addition[3]+"/"+datas[addition[3]][0])#, target_size=(64,64))
#input_shape = np.shape(temp_img)

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
model.add(Dense(3))       # クラスは2個
model.add(Activation('softmax'))

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# 実行。出力はなしで設定(verbose=0)。
history = model.fit(train_x, train_y, batch_size=5, epochs=20,
                    validation_data=(test_x, test_y), verbose=2)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.savefig("Img/my_first.png")

predict_classes = model.predict_classes(test_x)
# マージ。yのデータは元に戻す
mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(test_y, axis=1)})

# confusion matrix
pd.crosstab(mg_df['class'], mg_df['predict'])
plot_model(model, to_file="Img/model.png", show_shapes=True)
# モデルの保存
open('./Models/3class.json',"w").write(model.to_json())

# 学習済みの重みを保存
model.save_weights('./Models/3class.h5')
