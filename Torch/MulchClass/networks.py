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
    ##  input size (20 x 94 x 1)
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

class ArakiSoundNet1d(nn.Module):
    ##  input size (20 x 94)
    ##  48000Hz about 1sec wav file  || 44100Hz wav-file on mfcc -> (20 x 87)
    ##  output (11)
    def __init__(self):
        super(ArakiSoundNet1d, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(20, 32, 3, padding=1, stride=1),
            ########inpt,out,kernel
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, padding=0),
            nn.Conv1d(32, 64, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, padding=0),
            nn.Conv1d(64, 64, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, padding=0),
            nn.Conv1d(128, 256, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(11*512, 2048),
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

class TwostreamNet(nn.Module):
    ## input (224 x 224 x 3) (224 x 224 x 3)
    def __init__(self):
        super(TwostreamNet, self).__init__()
        self.features_1 = nn.Sequential(
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
         
        self.features_2 = nn.Sequential(
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
            nn.Linear(7*7*512*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,11))

    def forward(self, x, y):
        x = self.features_1(x)
        y = self.features_2(y)
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        z = torch.cat((x, y), 1)
        z = self.classifier(z)


        return z


class SoundBased_TwostreamNet(nn.Module):
    ## input  image:(224 x 224 x 3) sound:(20 x 94 x 1)
    def __init__(self):
        super(SoundBased_TwostreamNet, self).__init__()
        self.features_image = nn.Sequential(
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
         
        self.features_sound = nn.Sequential(
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
            nn.Linear(7*7*512*2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,11))
        self.fc = nn.Linear(512*2*11 , 512)
        
    def forward(self, images, sounds):
        x = self.features_image(images)
        y = self.features_sound(sounds)
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        y = y.reshape(len(y), 1, 1, 512) ## (batch, 512) --> (batch, 1, 1, 512)
        y = y.repeat(1,7,7,1)   ## (batch, 1, 1, 512) --> (batch, 7, 7, 512)
        y = y.view(y.size(0), -1)
        z = torch.cat((x, y), 1)
        z = self.classifier(z)

        return z


class SoundBased_ThreestreamNet(nn.Module):
    ## input  still_image:(224 x 224 x 3)
    ##        optic_image:(224 x 224 x 3)
    ##          wav_sound:( 20 x  94 x 1)
    def __init__(self):
        super(SoundBased_TwostreamNet, self).__init__()
        self.features_still = nn.Sequential(
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
        
        self.features_optic = nn.Sequential(
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
         
        self.features_sound = nn.Sequential(
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
            nn.Linear(7*7*512*3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,11))
        self.fc = nn.Linear(512*2*11 , 512)
        
    def forward(self, still, optic, sounds):
        stl = self.features_image(still)
        opt = self.features_image(optic)

        snd = self.features_sound(sounds)
        snd = snd.view(y.size(0), -1)
        snd = self.fc(snd)
        snd = snd.reshape(len(snd), 1, 1, 512) ## (batch, 512) --> (batch, 1, 1, 512)
        snd = snd.repeat(1,7,7,1)   ## (batch, 1, 1, 512) --> (batch, 7, 7, 512)

        snd = snd.view(snd.size(0), -1)
        opt = opt.view(opt.size(0), -1)
        stl = stl.view(stl.size(0), -1)
        
        z = torch.cat((stl, opt, snd), 1)
        z = self.classifier(z)

        return z
