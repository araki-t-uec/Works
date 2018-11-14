#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import sys, os
from PIL import Image
import numpy as np


class MyDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        #pandasでcsvデータの読み出し
        #self.image_dataframe = pd.read_csv(csv_file_path)
        self.image_dataframe = []
        self.data_dir = data_dir
        class_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        for i in class_dirs:
            i_files = os.listdir(os.path.join(data_dir, i))
            for j in i_files:
                self.image_dataframe.append([j,i])
        #画像データへの処理
        #self.transform = transform
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        filename = self.image_dataframe[idx][0]
        label = self.image_dataframe[idx][1]
        img_path = os.path.join(self.data_dir, label, filename)
        # #画像の読み込み
        image = Image.open(img_path)
        # #画像へ処理を加える
        t = transforms.Resize((224, 224))
        image = t(image)
        if self.transform:
            image = self.transform(image)
        # return image, label
        return image, int(label)-1


class Crop224(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        image = cv2.resize(image, (224,224))
        
        return image
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    
# uecfood= MyDataset("./UECFOOD100", Crop())
# dataloader = DataLoader(uecfood, batch_size=100,
#                         shuffle=True, num_workers=1)

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched) #['image'].size(),sample_batched['landmarks'].size())
