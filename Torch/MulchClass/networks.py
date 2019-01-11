#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import torch #基本モジュール
from torch.autograd import Variable #自動微分用
import torch.nn as nn #ネットワーク構築用
import torch.optim as optim #最適化関数
import torch.nn.functional as F #ネットワーク用の様々な関数
import torch.utils.data #データセット読み込み関連
import torchvision #画像関連
from torchvision import datasets, models, transforms #画像用データセット諸々

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x

    
class ArakiSoundNet(nn.Module):
    ##  input size (20 x 92 x 1)
    ##  48000Hz about 1sec wav file  || 44100Hz wav-file on mfcc -> (20 x 87 x 1)
    ##  output (11)
    def __init__(self):
        super(ArakiSoundNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=1),
            ########inpt,out,kernel
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(32, 64, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(128, 256, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*2*11, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,11)            
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# class SoundforVGG(nn.Module):
#     def __init__(self):
#         super(SoundforVGG, self).__init__()
#         self.forgray = nn.Sequential(
#             nn.Conv1d(20, 28, 3, padding=1, stride=1),
#             nn.MaxPool1d(2, padding=1),
#             nn.Conv1d(28, 44, 3, padding=1, stride=1)
#             #nn.MaxPool2d(2, padding=0)
#             #nn.Conv1d(56, 112, 3, padding=1, stride=1)
#             # nn.Conv1d(112, 224, 3, padding=1, stride=1)
#         )
#         self.forvgg = nn.Sequential(
#             #nn.Conv2d(3, 64, 3, padding=1),
            
#         )
#     def forward(self, x):
#         x = self.forgray(x)
#         x = x.unsqueeze(1)
#         x = self.forvgg(x)
#         return x

    

# class VGG16old(nn.Module):
#      # input size 224 x 224 x 3
#     def __init__(self):
#         super(VGG16old, self).__init__()
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
#         self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
#         self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
#         self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
#         self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
#         self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
#         self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
#         self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
#         self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
#         self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
#         self.fc1 = nn.Linear(7*7*512, 4096)
#         self.fc2 = nn.Linear(4096, 4096)
#         self.fc3 = nn.Linear(4096, 1000)
#         self.fc4 = nn.Linear(1000, 100)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = self.pool(F.relu(self.conv4(x)))
#         x = F.relu(self.conv5(x))
#         x = F.relu(self.conv6(x))
#         x = self.pool(F.relu(self.conv7(x)))
#         x = F.relu(self.conv8(x))
#         x = F.relu(self.conv9(x))
#         x = self.pool(F.relu(self.conv10(x)))
#         x = F.relu(self.conv11(x))
#         x = F.relu(self.conv12(x))
#         x = self.pool(F.relu(self.conv13(x)))
#         x = x.view(x.size(0),-1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.softmax(self.fc4(x))
#         return x

    
class VGG16(nn.Module):
    # input size 224 x 224 x 3
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=0),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=0))
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,1000))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

