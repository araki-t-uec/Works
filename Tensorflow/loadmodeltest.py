#!/usr/bin/python3
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



save_dir = "/home/yanai-lab/araki-t/www/Img/Works/"
save_file = "out.jpg"
save_path = save_dir + save_file
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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


def main():

    mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # モデルの読み込み
    model = keras.models.model_from_json(open('./Models/and.json', 'r').read())

    # 重みの読み込み
    model.load_weights('./Models/and.h5')

    # 読み込んだ学習済みモデルで予測

    predictions = model.predict(test_images)
    print(np.argmax(predictions[0]))
    print(test_labels[0])



    i = 12
    plt.figure(figsize=(6,3))
    
    plt.subplot(1,2,1)
    plot_image(i, predictions, test_labels, test_images)
    plt.savefig(save_path)

    plt.subplot(1,2,2)
    plot_value_array(i, predictions,  test_labels)
    plt.savefig(save_dir+"out3.jpg")

if __name__ == '__main__':
    main()
