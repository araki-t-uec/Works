#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import sys, os
import numpy as np

import dataload
import networks


epoch = 100
batch_size = 10
works = 4
learning_rate = 0.001

os.environ["CUDA_VISIBLE_DEVICES"]="1" #,2,3"
device = 'cuda' if t.cuda.is_available() else 'cpu'
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

datadir = "./data/UECFOOD100"
uecfood= dataload.MyDataset(datadir, transform)
train_size = int(0.8 * len(uecfood))
test_size = len(uecfood)-train_size  
train, test = t.utils.data.random_split(uecfood, [train_size, test_size])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=works)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=works)



model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096,100)
model = model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

def train(train_loader):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # for param in model.parameters():
        #   print(param.__class__.__name__)
        #   print(param.data)

        #
    #####
    train_loss = running_loss / len(train_loader)
    return train_loss



def test(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with t.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = outputs.max(1, keepdim=True)[1]
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            total += labels.size(0)
            
    val_loss = running_loss / len(test_loader)
    val_acc = correct / total
    
    return val_loss, val_acc



loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(epoch):
    loss = train(train_loader)
    val_loss, val_acc = test(test_loader)

    print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, val_loss, val_acc))
    
    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

print('Finished Training')
x = []
for i in range(0, len(loss_list)):
    x.append(i)
x = np.array(x)
plt.plot(x, np.array(loss_list), label="train")
plt.plot(x, np.array(val_loss_list), label="test")
#plt.plot(x, np.array(val_acc_list), label="acc")
plt.legend() # 凡例
plt.xlabel("epoch")
plt.ylabel("score")
plt.savefig('figurefinetune.png')
print("save to ./figurefinetune.png")

dataiter = iter(test_loader)
images, labels = dataiter.next()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

IMAGE_PATH = "."
#torchvision.utils.save_image(self.denorm(A.data.cpu()), '{}/real_{}.png'.format(IMAGE_PATH, j+1))
save_file = '{}/finetune{}e_{}b.png'.format(IMAGE_PATH, epoch, batch_size)
torchvision.utils.save_image(denorm(images.data.cpu()), save_file)


print("save to ", save_file)
print(labels)

t.save(model.state_dict(), './Models/finetuned.ckpt')
print("save to ./Models/finetuned.ckpt")
