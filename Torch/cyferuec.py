#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
from PIL import Image
import dataload
import networks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"]="0" #,2,3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch = 1
epoch = 20
works = 1
lr = 0.01
data_dir="./data/UECFOOD100"
uecfood = dataload.Cifarset(data_dir, transform=transform)
train_size = int(0.8 * len(uecfood))
test_size = len(uecfood)-train_size  
train_data, test_data = torch.utils.data.random_split(uecfood, [train_size, test_size])

trainloader =  torch.utils.data.DataLoader(train_data, batch_size=batch,
                                          shuffle=True, num_workers=works)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch,
                                          shuffle=True, num_workers=works)

classes = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]


model = networks.Cifar().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


def train(train_loader):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

    train_loss = running_loss / len(train_loader)
    return train_loss




def test(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
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


loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(epoch):  # loop over the dataset multiple times
    loss = train(trainloader)
    val_loss, val_acc = test(testloader)

    print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, val_loss, val_acc))
    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)


    
print('Finished Training')
#dataiter = iter(testloader)
#images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))

#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

torch.save(model.state_dict(), './Models/uecifer.ckpt')
print("save to ./Models/uecifer.ckpt")
