#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os, cv2

imaged = cv2.imread("./data/ImageNet/train/n02487347/n02487347_1956.JPEG")
#imaged = cv2.imread("./data/ImageNet/train/n02487347/n02487347_195.JPEG")
print(imaged.shape)
filename="./data/ImageNet/train/n01669191/n01669191_10054.JPEG"

image = cv2.imread(filename)
#if type(image) == type(imaged):
print(filename)
print(image.shape)


# data_dir = "./data/ImageNet/train/"
# class_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
# for i in class_dirs:
#     print(i)
#     i_files = os.listdir(os.path.join(data_dir, i))
#     for j in i_files:
#         filename = os.path.join(data_dir, i, j)
#         image = cv2.imread(filename)
#         if type(image) == type(imaged):
#             print(filename)

#
# n01669191
