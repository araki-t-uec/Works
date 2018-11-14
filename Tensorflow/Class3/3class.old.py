#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import os, sys
import tensorflow as tf
import tensorflow.python.platform
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import random


datadir = "Make3class/"
dirs = os.listdir(datadir)
#addition = [f for f in dirs if os.path.isdir(os.path.join(datadir, f))]
addition=["top","middle","bottom"] #,"test"]
classes=3
datas={}
train_x,train_y = [],[]
epoch = 20
batch = 4

if __name__ == '__main__':

    datadir = "Make3class/"
    dirs = os.listdir(datadir)
    #addition = [f for f in dirs if os.path.isdir(os.path.join(datadir, f))]
    addition=["top","middle","bottom"] #,"test"]
    classes=3
    datas={}
    train_x,train_y = [],[]
    # epoch = 20
    # batch = 150  

    for i in range(0,len(addition)):
        path = datadir+addition[i]
        files = os.listdir(path)
        datas[addition[i]]=([f for f in files if os.path.isfile(os.path.join(path, f))])
        if i < 3 :
            for j in datas[addition[i]]:
                #img = img_to_array(load_img(os.path.join(path,j)))#, target_size=(64,64)))
                img = cv2.imread(os.path.join(path,j))#, target_size=(64,64)))
                img = cv2.resize(img, (28,28))
                train_x.append(img)
                train_y.append(i) #tag number(top,middle,bottom )

    train_x = np.asarray(train_x) # list --to--> numpy.ndarray
    train_y = np.asarray(train_y)

    train_x = train_x.astype('float32')
    train_x = train_x / 255 # 画素値を0から1の範囲に変換

    train_y = np.identity(classes)[train_y]

    # 学習用データとテストデータ
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33)#, random_state=111) #rand seed
    #train_image, test_image, train_label, test_label = train_test_split(train_x, train_y, test_size=0.33)#, random_state=111) #rand seed
x=tf.placeholder(tf.float32)



#重み作る
def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name=name)
#バイアス作る
def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)
#畳み込み演算
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#プーリング演算
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#一つ目のweight,bias,5*5のフィルター,RGB3,出力32
W_conv1 = weight_variable([5, 5, 3, 32],"W_conv1")
b_conv1 = bias_variable([32],"b_conv1")

#畳み込み1、プーリング1
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#二つ目のweight,bias
W_conv2 = weight_variable([5, 5, 32, 64],"W_conv2")
b_conv2 = bias_variable([64],"b_conv2")

#畳み込み2、プーリング2
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#三つめのweight,bias
W_fc1 = weight_variable([7 * 7 * 64, 1024],"W_fc1")
b_fc1 = bias_variable([1024],"b_fc1")

#全結合層　一次元に直す
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#ドロップアウト
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#四つ目のweight,bias
W_fc2 = weight_variable([1024, classes],"W_fc2")
b_fc2 = bias_variable([classes],"b_fc2")

#出力
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#正解ラベル
right_rabel=tf.placeholder(tf.float32)

#損失関数
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=right_rabel, logits=y_conv))

#交差エントロピーを減らす
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#学習する
print()
tr=tf.placeholder(tf.float32)

#正解ラベルとの一致を調べる
correct_prediction=tf.equal(tf.argmax(tr,1),tf.argmax(y_conv,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#バッチ入れる箱
batch=[]
batch_rabels=[]

sess=tf.InteractiveSession()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(int(300)):
    #バッチを作る
    for a in range(20):
        b=random.randint(0,len(train_x)-1)
        batch.append(train_x[b])
        batch_rabels.append(train_y[b])
    #学習実行
    sess.run(train_step,feed_dict={x:batch,right_rabel:batch_rabels,keep_prob:0.5})
    if(i%100==0):
        #出力を出力
        print(sess.run(y_conv,feed_dict={x:test_x,right_rabel:test_y,keep_prob:1.0}))
    #テストデータで確認する

    if(i%10==0):
        print([i, int(1000)-1])
        print(sess.run(accuracy,feed_dict={x:test_x,tr:test_y,keep_prob:1.0}))

    #最終的な精度

    if(i==int((1000)-1)):
        print(sess.run(accuracy, feed_dict={x: test_x, tr: test_y, keep_prob: 1.0}))

#学習済みデータを保存

saver=tf.train.Saver()
saver.save(sess, "./model4.ckpt")
