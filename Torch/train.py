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
import torchvision.transforms as transforms
import torchvision.models as models
import sys, os
import numpy as np
import dataload
import networks


os.environ["CUDA_VISIBLE_DEVICES"]="0" #,2,3"

epochs = 200
batch_size = 10
works = 8
learning_rate = 0.0001
texts = "{}epoch, {}batch, {}num_works, lr={}"
print(texts.format(epochs, batch_size, works, learning_rate))

device = 'cuda' if t.cuda.is_available() else 'cpu'
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

datadir = "./data/Dog_opt_mean/"
#datadir = "./data/UECFOOD100/"
#dataset = dataload.MyDataset(datadir, transform)
dataset = dataload.Cifarset(datadir, transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset)-train_size  
train, test = t.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=works)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=works)


classes = {'bark':1,'cling':2,'command':3,'eat-drink':4,'look_at_handler':5,'run':6,'see_victim':7,'shake':8,'sniff':9,'stop':10,'walk-trot':11}


classes_num = 11
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, classes_num)
#print(model)
#print(model.classifier[6])
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

# inputs = t.randn(1, 3, 224, 224)
# inputs = inputs.to(device)
# out = model(inputs)
# print(out)
# exit()

def train(train_loader):
    model.train()
    running_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        #images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
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
            #images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
            # images = Variable(images)
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


def exe(test_loader):
    model.eval()
    predicts = {}
    with t.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predicted = outputs.max(1, keepdim=True)[1]

            for i in range(labels.shape[0]):
                label = labels[i].data.cpu().item()
                try:    
                    predicts[label].append(predicted[i].data.cpu().item())
                except:
                    predicts[label] = []
                    predicts[label].append(predicted[i].data.cpu().item())
            #print(predicts[label])

    return predicts




loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(epochs):
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
plt.savefig('./Img/figurefinetuneopt.png')
print("save to ./Img/figurefinetuneopt.png")

dataiter = iter(test_loader)
images, labels = dataiter.next()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

IMAGE_PATH = "./Img/"
#torchvision.utils.save_image(self.denorm(A.data.cpu()), '{}/real_{}.png'.format(IMAGE_PATH, j+1))
save_file = '{}{}_{}e_{}b.png'.format(IMAGE_PATH, labels, epoch, batch_size)
torchvision.utils.save_image(denorm(images.data.cpu()), save_file)

print("save to ", save_file)



def draw_heatmap(data, row_labels, column_labels):
    # 描画する
    fig, ax = plt.subplots()
    #heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
    heatmap = ax.pcolor(data, cmap=plt.cm.Reds)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    #plt.show()
    plt.savefig(IMAGE_PATH+'mean_heatmap_ndarray_opt.png')

    return heatmap

acc = exe(test_loader)
#print(acc)
arra = [0]*(classes_num) #]*(classes_num)
for i in acc:
    arra[i] = [0]*(classes_num)
    for j in acc[i]:
        arra[i][j]+=1

        
#arra = np.array(arra)
print(np.array(arra))
## Softmax
for i in range(len(arra)):

    total = sum(arra[i])
    for j in range(len(arra[i])):
        arra[i][j] = arra[i][j]/total
#print(arra)

arra = np.array(arra)
#print(acc)
print(arra)



classes = ['bark','cling','comand','eat','handlr','run','victim','shake','sniff','stop','walk']
draw_heatmap(arra, classes, classes)

# t.save(model.state_dict(), './Models/dogmean.ckpt')
# print("save to ./Models/dogmean.ckpt")
