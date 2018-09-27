#!/usr/bin/python3
# -*- coding:utf-8 -*-


import tensorflow as tf
config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0", allow_growth=True))
tf.Session(config=config)
print(tf.__version__)

from tensorflow import keras
print(keras.__version__)
#from keras.utils import multi_gpu_model


#with tf.device('/device:GPU:2'):
#  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#  c = tf.matmul(a, b)
## Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
## Runs the op.
#print(sess.run(c))



#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(28, 28)),
#    keras.layers.Dense(128, activation=tf.nn.relu),
#    keras.layers.Dense(10, activation=tf.nn.softmax)
#])
#model = multi_gpu_model(model, gpus=2)

