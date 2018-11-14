#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"]="1" #,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=16)
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dataframe = []
        self.data_dir = data_dir
        class_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        for i in class_dirs:
            i_files = os.listdir(os.path.join(data_dir, i))
            for j in i_files:
                self.image_dataframe.append([j,i])

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


uecfood = MyDataset("./UECFOOD100", transform=transform)
train_size = int(0.8 * len(uecfood))
test_size = len(uecfood)-train_size  
uectrain, uectest = torch.utils.data.random_split(uecfood, [train_size, test_size])

uecloader =  torch.utils.data.DataLoader(uectrain, batch_size=4,
                                          shuffle=True, num_workers=2)
uectest = torch.utils.data.DataLoader(uectest, batch_size=4,
                                          shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

#trainloader= uecloader
#testloader = uectest


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
data_dir="UECFOOD100"
classes = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    plt.savefig('figures.png')
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('figure.png')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
print(type(images))
print(images.shape)
print(labels)
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        # x = x.view(-1, 16*53*53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)
#print(net)
#exit()
#if device == "cuda":
    #net = torch.nn.DataParallel(net)
    #cudnn.benchmark = True

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



def train(train_loader):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
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

def valid(test_loader):
    net.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            #images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
            # images = Variable(images)
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            
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
for epoch in range(20):  # loop over the dataset multiple times
    loss = train(trainloader)
    val_loss, val_acc = valid(testloader)

    print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, val_loss, val_acc))
    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    
print('Finished Training')

#dataiter = iter(testloader)
#images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))

#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

