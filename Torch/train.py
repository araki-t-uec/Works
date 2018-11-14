#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader    
import sys, os, cv2

import numpy as np
import tutorial as tu
import networks

num_epochs = 1
batch_size = 10
learning_rate = 0.001

device = t.device("cuda" if t.cuda.is_available() else "cpu") #for GPU

uecfood= tu.MyDataset("./UECFOOD100", tu.Crop224())
train_size = int(0.8 * len(uecfood))
test_size = len(uecfood)-train_size  
train, test = t.utils.data.random_split(uecfood, [train_size, test_size])
train_loader = tu.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = tu.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=4)
#dataloader = tu.DataLoader(uecfood, batch_size=10, shuffle=True, num_workers=4)



model = networks.VGG16().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
#print(model)
#inputs = t.randn(1, 3, 224, 224)
#out = model(inputs)
#print(out)

def train(train_loader):
    model.train()
    running_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # for im in images:
        #     im = np.asarray(im)
        #     print(im.shape)
        #     cv2.imwrite('red.jpg', im)
        #images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
        images = images.to(device=device, dtype=t.float)
        labels = labels.to(device)
        #print(batch_idx, sample_batched) #['image'].size(),sample_batched['landmarks'].size())
        #optimizer.zero_grad()
        outputs = model(images)
        print(labels)

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        print("train: ", loss)
        #print(dir(model.fc4.named_parameters))
        #print(model.fc4[0])
        #print(model.fc4[0].weight.shape)
        #print(model.fc4.modules)
        #print(model.parameters()[0].data)
        # for param in model.parameters():
        #   print(param.__class__.__name__)
        #   print(param.data)

        loss.backward()
        optimizer.step()
        #
    #####
        
    train_loss = running_loss / len(train_loader)
    return train_loss



def valid(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with t.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
            # images = Variable(images)
            images = images.to(device=device, dtype=t.float)
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
for epoch in range(num_epochs):
    loss = train(train_loader)
    val_loss, val_acc = valid(test_loader)

    print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, val_loss, val_acc))
    
    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
