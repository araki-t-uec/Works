#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from keras.utils import multi_gpu_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="1", allow_growth=True))
#tf.Session(config=config)

#import cv2


mnist = tf.keras.datasets.fashion_mnist
#save_dir = "~/www/Img/Works/"
save_dir = "/home/yanai-lab/araki-t/www/Img/Works/"
save_file = "out.jpg"
save_path = save_dir + save_file
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

#plt.figure()
#cv2.imwrite(n_images[0])
#plt.colorbar()
#plt.grid(False)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.savefig(save_dir+"foriinrenge.jpg")
    plt.xlabel(class_names[train_labels[i]])


plt.savefig(save_path)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.compile(optimizer=tf.train.AdamOptimizer(),
#              loss='sparse_categorical_crossentropy')
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

# モデルの保存
open('and.json',"w").write(model.to_json())

# 学習済みの重みを保存
model.save_weights('and.h5')


#print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
#print(predictions[0])

#print(np.argmax(predictions[0]))
#print(test_labels[0])

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.savefig(save_path)#, img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


i = 12 
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.savefig(save_path)

plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.savefig(save_dir+"out2.jpg")



