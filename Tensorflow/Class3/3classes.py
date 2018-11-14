#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys, os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

dirname = "./Make3class/"
#classes = [f for f in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, f))]
classes = ["top","middle","bottom"]
class_n = len(classes)

trains = []
labels = []

for i in range(class_n):
    dirpath = os.path.join(dirname, classes[i])
    images = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    for j in images:
        image = cv2.imread(os.path.join(dirpath,j))
        trains.append(image)
        labels.append(i)
trains = np.asarray(trains)
trains = trains.astype('float32')
trains = trains / 255
labels = np.identity(class_n)[labels]

train_x, test_x, train_y, test_y = train_test_split(trains, labels, test_size=0.33)#, random_state=111) #rand seed
x=tf.placeholder(tf.float32)


#重み作る
def weight_variable(shape,name):
    print(shape)
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

#一つ目のweight,bias,3*3のフィルター,RGB=3,出力10
W_conv1 = weight_variable([3, 3, 3, 10],"W_conv1")
b_conv1 = bias_variable([10],"b_conv1")

#畳み込み1、プーリング1
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#二つ目のweight,bias
W_conv2 = weight_variable([2, 2, 10, 20],"W_conv2")
b_conv2 = bias_variable([20],"b_conv2")

#畳み込み2、プーリング2
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#三つめのweight,bias
#W_fc1 = weight_variable([2*2*20, 20],"W_fc1")
W_fc1 = weight_variable([180, 20],"W_fc1")
b_fc1 = bias_variable([20],"b_fc1")

#全結合層　一次元に直す
#h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*20])
h_pool2_flat = tf.reshape(h_pool2, [-1, 180])
#h_pool2_flat = tf.contrib.layers.flatten(h_pool2)
print(h_pool2_flat.shape)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#ドロップアウト
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#四つ目のweight,bias
W_fc2 = weight_variable([20, class_n],"W_fc2")
b_fc2 = bias_variable([class_n],"b_fc2")


h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#正解ラベル
right_rabel=tf.placeholder(tf.float32)

#損失関数
cross_entropy = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=right_rabel, logits=y_conv))
    tf.nn.softmax_cross_entropy_with_logits(labels=right_rabel, logits=h_fc2))

#交差エントロピーを減らす
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#学習する
tr=tf.placeholder(tf.float32)

#正解ラベルとの一致を調べる
#correct_prediction=tf.equal(tf.argmax(tr,1),tf.argmax(y_conv,1))
correct_prediction=tf.equal(tf.argmax(tr,1),tf.argmax(h_fc2,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#バッチ入れる箱
batch=[]
batch_rabels=[]

sess=tf.InteractiveSession()
init=tf.global_variables_initializer()
sess.run(init)


for i in range(int(100)):
    #バッチを作る
    for a in range(10):
        b=random.randint(0,len(train_x)-1)
        batch.append(train_x[b])
        batch_rabels.append(train_y[b])
    #学習実行
    sess.run(train_step,feed_dict={x:batch,right_rabel:batch_rabels,keep_prob:0.5})
    if(i%10==0):
        #出力を出力
#        print(sess.run(y_conv,feed_dict={x:test_x,right_rabel:test_y,keep_prob:1.0}))
        print(sess.run(h_fc2,feed_dict={x:test_x,right_rabel:test_y,keep_prob:1.0}))

    #テストデータで確認する

    if(i%10==0):
        print([i, int(100)-1])
        print(sess.run(accuracy, feed_dict={x:test_x, tr:test_y,keep_prob:1.0}))

    #最終的な精度

    if(i==int((100)-1)):
        print(sess.run(accuracy, feed_dict={x:test_x, tr:test_y, keep_prob: 1.0}))

#学習済みデータを保存

saver=tf.train.Saver()
saver.save(sess, "./model4.ckpt")
print("./model4.ckpt is written\n")
