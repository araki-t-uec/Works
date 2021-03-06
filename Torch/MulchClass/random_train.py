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

from opts import parse_opts


opt = parse_opts()
device = 'cuda' if t.cuda.is_available() else 'cpu'
if opt.mulch_gpu == False:
    gpu_num = opt.gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num) #,2,3"

epochs = opt.epochs
batch_size = opt.batch_size 
works = opt.num_works
learning_rate = opt.lr
threthold = opt.threthold
IMAGE_PATH = "./Img/"
result_path = opt.result_path
annotation_test = opt.annotation_file
#annotation_test = opt.annotation_file+".txt"
#annotation_train = opt.annotation_file+"V.txt"
corename = opt.save_name+"_"+opt.annotation_file.split("/")[-1]+"_th-"+str(int(threthold*10))+"_lr-"+str(str(int(learning_rate**(-1))).count("0"))
texts = "{}epoch, {}batch, {}num_works, lr={}, threthold={}"
print(corename)
print(texts.format(epochs, batch_size, works, learning_rate, threthold))

# transform_train = transforms.Compose(
#     [transforms.Resize(224),
#      transforms.Pad(16),
#      transforms.RandomCrop((224)),
#      transforms.ToTensor(),
#      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # imagenet
#      ])
transform_test = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # imagenet
    ])

data_dir = "./Resque/Labeled/NoLabeled/"
#annotation_file ="./Resque/Labeled/NoLabeled/annotation.txt"
#annotation_file ="./Tmptest/annotation.txt"
classes = {'bark':0,'cling':1,'command':2,'eat-drink':3,'look_at_handler':4,'run':5,'see_victim':6,'shake':7,'sniff':8,'stop':9,'walk-trot':10}


## Model
classes_num = 11
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, classes_num)
#print(model)
#print(model.classifier[6])
if opt.mulch_gpu == "True":
    print("Let's use", t.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model) # make parallel
    #cudnn.benchmark = True
model = model.to(device)


## Load dataset.
#train_dataset = dataload.MulchVideoset(annotation_train, data_dir, classes, transform_train)
#test_dataset = dataload.MulchVideoset(annotation_test, data_dir, classes, transform_test)
dataset = dataload.MulchVideoset(annotation_test, data_dir, classes, transform_test)

train_size = int(0.8 * len(dataset))
test_size = len(dataset)-train_size  
train, test = t.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=works)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=works)




criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)


def display(train_loader, namae):
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        
        # if title is not None:
        #     plt.title(title)
        # plt.pause(0.001)  # pause a bit so that plots are updated
        plt.imsave(namae+'.jpg', inp)
    # Get a batch of training data
    inputs, labels = next(iter(train_loader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out)

# display(train_loader, opt.annotation_file.split("/")[-1])
display(train_loader, corename)

def nofk(output, gt_labels, threthold=0.5):
    return(np.where(output < threthold, 0, 1))

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
        
        #loss = criterion(outputs, labels)
        loss = 1*F.multilabel_soft_margin_loss(outputs, labels.cuda(non_blocking=True).float())

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
    relevant = [0]*classes_num # Count ground trues for calculate the Recall
    selected = [0]*classes_num # Count selected elements for calculate the Precision
    true_positives = [0]*classes_num  # Count true positive relevant for the Recall and Precision
    #recall = 0
    #total = 0
    with t.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            #images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
            # images = Variable(images)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            #loss = criterion(outputs, labels)
            loss = 1*F.multilabel_soft_margin_loss(outputs, labels.cuda(non_blocking=True).float())
            running_loss += loss.item()

            labels = labels.cpu().numpy()
            sigmoided = F.sigmoid(outputs)

            predicted = nofk(sigmoided, labels, threthold=threthold)
            #print((labels * predicted).tolist())
            #true_positives.extend((labels * predicted).tolist())
            relevant += np.sum(labels, axis=0)
            selected += np.sum(predicted, axis=0)
            true_positives += np.sum((labels * predicted), axis=0)
    ##
    precision = true_positives/relevant
    recall = true_positives/selected
    val_loss = running_loss / len(test_loader)
    
    return val_loss, precision, recall


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
            print(predicts[label])

    return predicts




loss_list = []
val_loss_list = []
precision_list = []
recall_list = []
for epoch in range(epochs):
    loss = train(train_loader)
    val_loss, precision, recall = test(test_loader)

    print('epoch %d, loss: %.4f, val_loss: %.4f, precision: %s, recall: %s' % (epoch, loss, val_loss, list(precision), list(recall)))
    
    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    precision_list.append(precision)
    recall_list.append(recall)

    
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
#plt.savefig('./Img/figurefinetuneopt.png')
#print("save to ./Img/figurefinetuneopt.png")
plt.savefig(os.path.join(IMAGE_PATH,corename+'.png'))
print("save to "+corename+".png")

### Save a model.
t.save(model.state_dict(), os.path.join(result_path+corename+'.ckpt'))
print("save to "+os.path.join(result_path+corename+'.ckpt'))


exit()


dataiter = iter(test_loader)
images, labels = dataiter.next()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

#torchvision.utils.save_image(self.denorm(A.data.cpu()), '{}/real_{}.png'.format(IMAGE_PATH, j+1))
save_file = os.path.join(IMAGE_PATH, corename+"_denorm.jpg")
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
    plt.savefig(IMAGE_PATH+corename+'_heatmap.png')

    return heatmap


## Make heatmap image.
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

