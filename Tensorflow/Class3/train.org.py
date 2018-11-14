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

datadir = "Make3class/"
dirs = os.listdir(datadir)
#addition = [f for f in dirs if os.path.isdir(os.path.join(datadir, f))]
addition=["top","middle","bottom"] #,"test"]
classes=3
datas={}
train_x,train_y = [],[]
epoch = 20
batch = 4

for i in range(0,len(addition)):
    path = datadir+addition[i]
    files = os.listdir(path)
    datas[addition[i]]=([f for f in files if os.path.isfile(os.path.join(path, f))])
    if i < 3 :
        for j in datas[addition[i]]:
            #img = img_to_array(load_img(os.path.join(path,j)))#, target_size=(64,64)))
            img = cv2.imread(os.path.join(path,j))#, target_size=(64,64)))
            train_x.append(img)
            train_y.append(i) #tag number(top,middle,bottom )

train_x = np.asarray(train_x) # list --to--> numpy.ndarray
train_y = np.asarray(train_y)

train_x = train_x.astype('float32')
train_x = train_x / 255 # 画素値を0から1の範囲に変換

train_y = np.identity(classes)[train_y]

# 学習用データとテストデータ
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33)#, random_state=111) #rand seed

NUM_CLASSES = 3
IMAGE_SIZE = 12
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

shape = np.asarray([IMAGE_SIZE,IMAGE_SIZE,3])
shapes = np.asarray([None, IMAGE_SIZE,IMAGE_SIZE,3])

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'train.txt', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', '/tmp/data', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

def inference(images_placeholder, keep_prob):
    """ 予測モデルを作成する関数

    引数: 
      images_placeholder: 画像のplaceholder
      keep_prob: dropout率のplace_holder

    返り値:
      y_conv: 各クラスの確率(のようなもの)
    """
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 畳み込み層の作成
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層の作成
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    
    # 入力を28x28x3に変形
    # x_image = tf.reshape(images_placeholder, [-1, 28, 28, 3])
    x_image = tf.reshape(images_placeholder, [-1, 10, 10, 3])

    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        # W_conv1 = weight_variable([5, 5, 3, 32])
        W_conv1 = weight_variable([3, 3, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    print("h conv1: ", h_conv1.shape)
    print("h pool1: ", h_pool1.shape)
    # 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        # W_conv2 = weight_variable([5, 5, 32, 64])
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    print("W conv2: ",W_conv2.shape)
    print("b conv2: ",b_conv2.shape)
    print("h conv2: ",h_conv2.shape)
    
    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
    print("h pool2: ", h_pool2.shape)
    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
#        W_fc1 = weight_variable([7*7*64, 1024])
        W_fc1 = weight_variable([3*3*64, 1024])
        b_fc1 = bias_variable([1024])
#        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#        h_fc1 = tf.nn.relu(tf.matmul(h_pool2, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print("W fc1: ", W_fc1.shape)
    print("b fc1: ", b_fc1.shape)
    print("h fc1: ", h_fc1.shape)
    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
    print("W fc2: ", W_fc2.shape)
    print("b fc2: ", b_fc2.shape)

    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    print("h fc1 drop: ", h_fc1_drop.shape)
    # 各ラベルの確率のようなものを返す
    print("y conv: ", y_conv.shape)
    print(np.asarray(y_conv))
    exit()
    return(y_conv)

def loss(logits, labels):
    """ lossを計算する関数

    引数:
      logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      cross_entropy: 交差エントロピーのtensor, float

    """

    # 交差エントロピーの計算
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    # TensorBoardで表示するよう指定
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    """ 訓練のOpを定義する関数

    引数:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数

    返り値:
      train_step: 訓練のOp

    """

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    """ 正解率(accuracy)を計算する関数

    引数: 
      logits: inference()の結果
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]

    返り値:
      accuracy: 正解率(float)

    """
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

if __name__ == '__main__':
    # # ファイルを開く
    
    # f = open(FLAGS.train, 'r')
    # # データを入れる配列
    # train_image = []
    # train_label = []
    # for line in f:
    #     # 改行を除いてスペース区切りにする
    #     line = line.rstrip()
    #     l = line.split()
    #     # データを読み込んで28x28に縮小
    #     img = cv2.imread(l[0])
    #     img = cv2.resize(img, (28, 28))
    #     # 一列にした後、0-1のfloat値にする
    #     train_image.append(img.flatten().astype(np.float32)/255.0)
    #     # ラベルを1-of-k方式で用意する
    #     tmp = np.zeros(NUM_CLASSES)
    #     tmp[int(l[1])] = 1
    #     train_label.append(tmp)
    # # numpy形式に変換
    # train_image = np.asarray(train_image)
    # train_label = np.asarray(train_label)
    # f.close()

    # f = open(FLAGS.test, 'r')
    # test_image = []
    # test_label = []
    # for line in f:
    #     line = line.rstrip()
    #     l = line.split()
    #     img = cv2.imread(l[0])
    #     img = cv2.resize(img, (28, 28))
    #     test_image.append(img.flatten().astype(np.float32)/255.0)
    #     tmp = np.zeros(NUM_CLASSES)
    #     tmp[int(l[1])] = 1
    #     test_label.append(tmp)
    # test_image = np.asarray(test_image)
    # test_label = np.asarray(test_label)
    # f.close()
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
                train_x.append(img)
                train_y.append(i) #tag number(top,middle,bottom )

    train_x = np.asarray(train_x) # list --to--> numpy.ndarray
    train_y = np.asarray(train_y)

    train_x = train_x.astype('float32')
    train_x = train_x / 255 # 画素値を0から1の範囲に変換

    train_y = np.identity(classes)[train_y]

    # 学習用データとテストデータ
    # train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33)#, random_state=111) #rand seed
    train_image, test_image, train_label, test_label = train_test_split(train_x, train_y, test_size=0.33)#, random_state=111) #rand seed

    with tf.Graph().as_default():
        # 画像を入れる仮のTensor
        #images_placeholder=tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        images_placeholder = tf.placeholder("float", shape=shapes)
        # ラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder("float")

        # inference()を呼び出してモデルを作る
        logits = inference(images_placeholder, keep_prob)
        # loss()を呼び出して損失を計算
        loss_value = loss(logits, labels_placeholder)
        # training()を呼び出して訓練
        train_op = training(loss_value, FLAGS.learning_rate)
        # 精度の計算
        acc = accuracy(logits, labels_placeholder)

        # 保存の準備
        saver = tf.train.Saver()
        # Sessionの作成
        sess = tf.Session()
        # 変数の初期化
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        # TensorBoardで表示する値の設定
        summary_op = tf.summary.merge_all()
#        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph_def)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        
        # 訓練の実行
        for step in range(FLAGS.max_steps):
            for i in range(len(train_image),FLAGS.batch_size):
                # batch_size分の画像に対して訓練の実行
                batch = FLAGS.batch_size*i
                # feed_dictでplaceholderに入れるデータを指定する
                sess.run(train_op, feed_dict={
                    images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                    images_placeholder: train_label[batch:batch+FLAGS.batch_size],
                    labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
                    keep_prob: 0.5})

            # 1 step終わるたびに精度を計算する
            train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            print("step %d, training accuracy %g"%(step, train_accuracy))

            # 1 step終わるたびにTensorBoardに表示する値を追加する
            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

    # 訓練が終了したらテストデータに対する精度を表示
    print("test accuracy %g"%sess.run(acc, feeddict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0}))

    # 最終的なモデルを保存
    save_path = saver.save(sess, "./Models/model.ckpt")
