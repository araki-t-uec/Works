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
from keras.regularizers import l2
from keras.layers import Input, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
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
epoch = 2
batch = 15 
img_height,img_width=240,320
savename = "AlexNet_finetune"
#savename = "plt_test"

train_x = np.load("dog_x.npy")
train_y = np.load("dog_y.npy")
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.20)#, random_state=111) #rand seed

#モデルを作ってください．
model = Sequential()
l2_reg=0
img_shape=(img_height, img_width, 3)


alexnet = Sequential()

# Layer 1
alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
                padding='same', kernel_regularizer=l2(l2_reg)))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
alexnet.add(Conv2D(256, (5, 5), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(512, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(1024, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))

# Layer 5
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(1024, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
alexnet.add(Flatten())
alexnet.add(Dense(3072))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.5))

# Layer 7
alexnet.add(Dense(4096))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.5))

# Layer 8
alexnet.add(Dense(class_num))
alexnet.add(BatchNormalization())
alexnet.add(Activation('softmax'))
######## Keras Version problem =======
# alexnet.load_weights("./Models/alexnet_weights.h5")

# # 最後のconv層の直前までの層をfreeze
# for layer in alexnet.layers[:4]:
#     layer.trainable = False
#======================================


# 多クラス分類を指定
alexnet.compile(loss='categorical_crossentropy',
          optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
          metrics=['accuracy'])

#plot_model(resnet.model, to_file="Img/Models/"+savename+"_model.png", show_shapes=True)

secc = time()
#keras.backend.get_session().run(tf.initialize_all_variables())
#keras.backend.get_session().run(tf.global_variables_initializer())
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
history = alexnet.fit(train_x, 
                      train_y, 
                      batch_size=batch, 
                      epochs=epoch,
                      verbose=2,
                      validation_data=(test_x, test_y)) 
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
open('./Models/'+savename+'.json',"w").write(alexnet.to_json())
# 学習済みの重みを保存
alexnet.save_weights('./Models/'+savename+'.h5')


# テストデータに適用
predict_classes = alexnet.predict_classes(test_x)

# マージ。yのデータは元に戻す
mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(test_y, axis=1)})
#mg_df = pd.DataFrame({'predict': np.argmax(predict_classes, axis=1), 'class': np.argmax(test_y, axis=1)})

# confusion matrix
hge = pd.crosstab(mg_df['class'], mg_df['predict'])

print(hge)
print("fitting time: ", cost)
