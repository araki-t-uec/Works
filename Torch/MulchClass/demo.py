#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import sys, os
import numpy as np
import dataload
import networks
import argparse
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    default='Results/resquedog_20160710_th-5_lr-7.ckpt',
    type=str,
    help='path of pre-trained model')
parser.add_argument(
    '--image',
    #        default='Resque/Labeled/NoLabeled',
    type=str,
    help='image to predict')
opts = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_PATH = "./Img/"

transform_test = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # imagenet
    ])

classes = {'bark':0,'cling':1,'command':2,'eat-drink':3,'look_at_handler':4,'run':5,'see_victim':6,'shake':7,'sniff':8,'stop':9,'walk-trot':10}


## Model
classes_num = 11
#model = models.vgg16(pretrained=False)
#model.classifier[6] = nn.Linear(4096, classes_num)
model = networks.VGG16()
model.classifier[6] = nn.Linear(4096, classes_num)
try:
    param = torch.load(opts.model)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in param.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

except:
    print("$ ./demo.py --model [trained_model] --image [image_path]")
    exit()
    
#model.load_state_dict(param)
model.load_state_dict(new_state_dict)
model = model.to(device)



def nofk(output, threthold=0.5):
    return(np.where(output < threthold, 0, 1))

def demo():
    model.eval()
    with torch.no_grad():
        image = Image.open(opts.image)
        image = transform_test(image)
        image = image.unsqueeze(0).to(device)
        output = model(image)
        sigmoided = F.sigmoid(output)
        predicted = nofk(sigmoided)

    ##
    return predicted


predict = demo()[0]
label = []
for i in range(len(predict)):
    if predict[i] == 1:
        label.extend([k for k, v in classes.items() if v == i])

print(label)
