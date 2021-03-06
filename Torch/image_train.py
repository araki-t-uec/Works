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

os.environ["CUDA_VISIBLE_DEVICES"]="0,1" #,2,3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

datadir = "./data/ImageNet/train/"
data_im = dataload.YourDataset(datadir, transform)
train_size = int(0.8 * len(data_im))
test_size = len(data_im)-train_size  
train_im, test_im = torch.utils.data.random_split(data_im, [train_size, test_size])
print("load image path: ", datadir)


batch_size = 10
num_workers= 8
epoch = 2
print(batch_size, "batch, ", epoch, "epoch")
trainloader =  torch.utils.data.DataLoader(train_im, batch_size= batch_size,
                                          shuffle=False, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(test_im, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

classes = [f for f in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, f))]

categories = {}
count = 0
for i in classes:
    if i not in categories.keys():
        categories[i] = count
        count += 1
        #print(count,": ",i) 
#print(categories)
model = networks.VGG16_1000().to(device)
#print(data_im.categories)
#print(model)
#if device == "cuda":
    #model = torch.nn.DataParallel(net)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(train_loader):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        print(labels)
        print(type(labels))
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
plt.plot(x, np.array(loss_list), label="loss")
plt.plot(x, np.array(val_loss_list), label="loss")
#plt.plot(x, np.array(val_acc_list), label="loss")
plt.legend() # 凡例
plt.xlabel("epoch")
plt.ylabel("score")
plt.savefig('figure_imagenet_.png')
print("save to ./figure_image_net.png")

dataiter = iter(testloader)
images, labels = dataiter.next()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

IMAGE_PATH = "."
#torchvision.utils.save_image(self.denorm(A.data.cpu()), '{}/real_{}.png'.format(IMAGE_PATH, j+1))
save_file = '{}/imnet_{}e_{}b.png'.format(IMAGE_PATH, epoch, batch_size)
torchvision.utils.save_image(denorm(images.data.cpu()), save_file)


print("save to ", save_file)
print(labels)

torch.save(model.state_dict(), './Models/imagenet_model.ckpt')
print("save to ./Models/imagenet_model.ckpt")
