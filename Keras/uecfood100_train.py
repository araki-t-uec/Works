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
import cv2
import time
#img = cv2.cvtColor(cv2.imread("/export/data/dataset/UECFOOD/UECFOOD100/79/8424.jpg"), cv2.COLOR_BGR2RGB)

#img = img_to_array(load_img("/export/data/dataset/UECFOOD/UECFOOD100/79/8424.jpg"))#8390.jpg"))#, target_size=(64,64))
#print(type(img), np.shape(img))
#print(type(im2), np.shape(im2))

#exit()

datadir="/export/data/dataset/UECFOOD/UECFOOD100/"
dirs = os.listdir(datadir)
addition = [f for f in dirs if os.path.isdir(os.path.join(datadir, f))]
datas = {}
train_x = [] 
train_y = []
size = (120,90)
epochs = 20
savename = "uecfood100_batchorg"
for i in range(0,len(addition)):
    print("load images from " + str(i) + ": "+addition[i])
    path = datadir+addition[i]
    files = os.listdir(path)
    #datas[addition[i]]=([f for f in files if os.path.isfile(os.path.join(path, f))])
    for f in files:
        try:
            img_path = os.path.join(path,f)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size)
            #img = img_to_array(load_img(img_path))#, target_size=(64,64))
            train_x.append(img)
            train_y.append(addition[i]) #tag number(top,middle,bottom )
        except:
            #print(f+" is not img file..") 
            pass

train_x = np.asarray(train_x) # list --to--> numpy.ndarray
train_y = np.asarray(train_y)

train_x = train_x.astype('float32')
train_x = train_x / 255 # 画素値を0から1の範囲に変換

classes = len(addition)

#train_y = np_utils.to_categorical(train_y, classes) #[0,1,2] -> [[1,0,0],[0,1,0],[0,0,1]]
tmp = []
for i in train_y:
    l = [0]*classes
    l[int(i)-1] = 1
    tmp.append(l)
train_y = np.asarray(tmp)
#print(train_y.shape)
#exit()

# 学習用データとテストデータ
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33)#, random_state=111) #rand seed


#temp_img = load_img(datadir+addition[3]+"/"+datas[addition[3]][0])#, target_size=(64,64))
#input_shape = np.shape(temp_img)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=train_x.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
#model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
#model.add(Activation('sigmoid'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes))  
model.add(Activation('softmax'))

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])
plot_model(model, to_file="Img/Models/"+savename+"_model.png", show_shapes=True)

# 実行。出力はなしで設定(verbose=0)。
secc = time.time()

history = model.fit(train_x, train_y, batch_size=8, epochs=epochs,
                    validation_data=(test_x, test_y), verbose=1)

print("fit time: ", time.time()-secc)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.savefig("Img/"+savename+"_acc.png")

predict_classes = model.predict_classes(test_x)
# マージ。yのデータは元に戻す
mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(test_y, axis=1)})

# confusion matrix
pd.crosstab(mg_df['class'], mg_df['predict'])
# モデルの保存
open('./Models/'+savename+'.json',"w").write(model.to_json())

# 学習済みの重みを保存
model.save_weights('./Models/'+savename+'.h5')
# plt.plot(range(1, epochs+1), history.history['acc'], label="training")
# plt.plot(range(1, epochs+1), history.history['val_acc'], label="validation")
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig("Img/uecfood100_train_data_30-30.png")
