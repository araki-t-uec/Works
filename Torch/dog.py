#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import os
import dataload
import networks
import cv2
os.environ["CUDA_VISIBLE_DEVICES"]="1" #,2,3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

datadir = "./data/Dogs/Movies"
classes = ['Car', 'Drink', 'Feed', 'LookLeft', 'LookRight', 'Pet', 'PlayBall', 'Shake', 'Sniff', 'Walk']
class_num = len(classes)
train_x = []
train_y = []
savename = "dog_mean"
#savename = "plt_test"

path = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
for j in path:
    print(j)
    n = 0
    frames = []
    path = os.path.join(datadir,j)
    #        print(path)
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    while(ret == True):
        #print("hoge")
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = img_to_array(frame)
        #frame = cv2.resize(frame, (24,24))
        frames.append(frame)
        ret, frame = cap.read()
        # cv2.imwrite('frame.png',gray)
    # frames = (np.asarray(frames).mean(axis=0)) # the average overall frames
    frames = np.asarray(frames)
    #train_x.append(frames)
    #train_y.append(i)
    cap.release()
    print(frames.shape)
#exit()


dataset = dataload.AVIload(datadir, transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset)-train_size  
traindata, testdata = torch.utils.data.random_split(dataset, [train_size, test_size])


batch_size = 1
num_workers= 2
epoch = 2
print(batch_size, "batch, ", epoch, "epoch")
trainloader =  torch.utils.data.DataLoader(traindata, batch_size= batch_size,
                                          shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

classes = [f for f in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, f))]

model = networks.VGG16old().to(device)
print(model)
#if device == "cuda":
    #model = torch.nn.DataParallel(net)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(train_loader):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        print(labels)
        print(labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()  
        #if i % 100 == 99 :
          #  print(running_loss / len(train_loader))
    train_loss = running_loss / len(train_loader)
    return train_loss


def test(test_loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = outputs.max(1, keepdim=True)[1]
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            total += labels.size(0)
            #print(labels)
            #print(predicted)
            #print(labels.view_as(predicted))
            #print(predicted.eq(labels.view_as(predicted)))
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
x = []
for i in range(0, len(loss_list)):
    x.append(i)
x = np.array(x)
plt.plot(x, np.array(loss_list), label="train")
plt.plot(x, np.array(val_loss_list), label="test")
plt.plot(x, np.array(val_acc_list), label="acc")
plt.legend() # 凡例
plt.xlabel("epoch")
plt.ylabel("score")
plt.savefig('figureuec.png')
print("save to ./figureuec.png")

dataiter = iter(testloader)
images, labels = dataiter.next()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

IMAGE_PATH = "."
#torchvision.utils.save_image(self.denorm(A.data.cpu()), '{}/real_{}.png'.format(IMAGE_PATH, j+1))
save_file = '{}/real_{}e_{}b.png'.format(IMAGE_PATH, epoch, batch_size)
torchvision.utils.save_image(denorm(images.data.cpu()), save_file)


print("save to ", save_file)
print(labels)

torch.save(model.state_dict(), './Models/model.ckpt')
print("save to ./Models/model.ckpt")
