#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
from PIL import Image
import networks, dataload



batch = 20
works = 16
epochs = 200
lr= 0.0001
save_dir = "./Img" 
os.environ["CUDA_VISIBLE_DEVICES"]="1" #,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True, num_workers=works)


# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.image_dataframe = []
#         self.data_dir = data_dir
#         class_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
#         for i in class_dirs:
#             i_files = os.listdir(os.path.join(data_dir, i))
#             for j in i_files:
#                 self.image_dataframe.append([j,i])

#         self.transform = transforms.Compose([transforms.ToTensor(),
#                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        
#     def __len__(self):
#         return len(self.image_dataframe)

#     def __getitem__(self, idx):
#         #dataframeから画像へのパスとラベルを読み出す
#         filename = self.image_dataframe[idx][0]
#         label = self.image_dataframe[idx][1]
#         img_path = os.path.join(self.data_dir, label, filename)
#         # #画像の読み込み
#         image = Image.open(img_path)
#         # #画像へ処理を加える
#         t = transforms.Resize((224, 224))
#         image = t(image)
#         if self.transform:
#             image = self.transform(image)
#         # return image, label
#         return image, int(label)-1

data_dir="./data/UECFOOD100"
uecfood = dataload.Cifarset(data_dir, transform=transform)
train_size = int(0.8 * len(uecfood))
test_size = len(uecfood)-train_size  
uectrain, uectest = torch.utils.data.random_split(uecfood, [train_size, test_size])

uecloader =  torch.utils.data.DataLoader(uectrain, batch_size=batch,
                                          shuffle=True, num_workers=works)
uectest = torch.utils.data.DataLoader(uectest, batch_size=batch,
                                          shuffle=True, num_workers=works)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

trainloader= uecloader
testloader = uectest
classes = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def imshow(img):
    plt.savefig('figures.png')
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    save_file = os.path.join(save_dir, "cifar.png")
    torchvision.utils.save_image(denorm(images.data.cpu()), save_file)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
print(type(images))
print(images[0][0][0][0].data.numpy())
print(images.shape)
print(labels)
print(len(classes))
#exit()
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
import torch.nn as nn
import torch.nn.functional as F

net = networks.Cifar(len(classes))
net = net.to(device)
#print(net)
#exit()
if device == "cuda":
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

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
        running_loss += loss.item()
        
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
for epoch in range(epochs):  # loop over the dataset multiple times
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
plt.savefig('figurecifsar.png')
print("save to ./figurecifar.png")

# print images
imshow(torchvision.utils.make_grid(images))

#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

torch.save(net.state_dict(), './Models/cyfer.ckpt')
print("save to ./Models/cyfer.ckpt")
