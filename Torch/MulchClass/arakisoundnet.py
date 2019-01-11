#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
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
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
if opt.mulch_gpu == False:
    gpu_num = opt.gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num) #,2,3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = opt.jpg_path
epochs = opt.epochs
batch_size = opt.batch_size 
works = opt.num_works
learning_rate = opt.lr
swing_rate = opt.swing_rate
swing_period = opt.swing_period
threthold = opt.threthold
IMAGE_PATH = "./Img/Losses"
result_path = opt.result_path
annotation_test = opt.annotation_file+"_test.txt"
annotation_train = opt.annotation_file+"_train.txt"
addtext = ""
if swing_rate != 1.0:
    addtext = "_sr-{}_sp-{}".format(int(swing_rate*10), swing_period)

corename = opt.save_name+"_"+opt.annotation_file.split("/")[-1]+"_bc-"+str(batch_size)+"_lr-"+str(str(int(learning_rate**(-1))).count("0"))+addtext
texts = "{}epoch, {}batch, {}num_works, lr={}, threthold={}"
print(corename)
print(texts.format(epochs, batch_size, works, learning_rate, threthold), data_dir)

# #annotation_file ="./Resque/Labeled/NoLabeled/annotation.txt"
# #annotation_file ="./Tmptest/annotation.txt"
classes = {'bark':0,'cling':1,'command':2,'eat-drink':3,'look_at_handler':4,'run':5,'see_victim':6,'shake':7,'sniff':8,'stop':9,'walk-trot':10}





# #### Test call mfcc
# import librosa
# wav_file = "./Resque/Sounds/1sec30f/20150801/20150801_000200.wav"
# #x, fs = librosa.load(wav_file, sr=44100)
# x, fs = librosa.load(wav_file, sr=48000)
# mfccs = librosa.feature.mfcc(x, sr=fs)
# print(x, fs)
# print(mfccs)
# print(mfccs.shape)
# mfccs = torch.Tensor(mfccs)
# mfccs = mfccs.unsqueeze(0)    ## [20, 87] -> [1,20,87]
# mfccs = torch.cat((mfccs, mfccs)) # 2 batchs
# print(mfccs.shape)
# ####
# exit()




## Model
classes_num = 11
model = networks.ArakiSoundNet()
#print(model)
if opt.mulch_gpu == "True":
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model) # make parallel
    #cudnn.benchmark = True
model = model.to(device)


## Load dataset.
train_dataset = dataload.DogSounds(annotation_train, data_dir, classes)
test_dataset = dataload.DogSounds(annotation_test, data_dir, classes)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=works)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=works)

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

#display(train_loader, corename)

def nofk(output, gt_labels, threthold=0.5):
    return(np.where(output < threthold, 0, 1))

def train(train_loader, learning_rate):
    model.train()
    running_loss = 0
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

    for batch_idx, (sounds, labels) in enumerate(train_loader):
        #images = images.transpose(1, 3) # (1,224,224,3) --> (1,3,224,224)
        sounds = sounds.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(sounds)
        
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
    with torch.no_grad():
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
    micro_precision = np.sum(true_positives)/np.sum(relevant)
    micro_recall = np.sum(true_positives)/np.sum(selected)
    val_loss = running_loss / len(test_loader)
    
    return val_loss, micro_precision, micro_recall, precision, recall


loss_list = []
val_loss_list = []
precision_list = []
recall_list = []
for epoch in range(epochs):
    loss = train(train_loader, learning_rate)
    val_loss, micro_precision, micro_recall, precision, recall = test(test_loader)

    print('epoch %d, loss: %.4f, val_loss: %.4f, micro_precision: %.4f, micro_recall: %.4f, precision: %s, recall: %s' % (epoch, loss, val_loss, micro_precision, micro_recall, list(precision), list(recall)))

    if epochs%(swing_period+1) == swing_period:
        learning_rate = learning_rate = swing_rate
        print("chenge learning_rate! to ", learning_rate)
    # logging
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    precision_list.append(micro_precision)
    recall_list.append(micro_recall)

    plt.figure()
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

    plt.figure()
    plt.plot(x, np.array(precision_list), label="precision", color="green")
    plt.plot(x, np.array(recall_list), label="recall", linestyle="dashed", color="purple")
    plt.legend() # 凡例
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(os.path.join(IMAGE_PATH,corename+'_accuracy.png'))


    
print('Finished Training')
plt.figure()
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
torch.save(model.state_dict(), os.path.join(result_path+corename+'.ckpt'))
print("save to "+os.path.join(result_path+corename+'.ckpt'))


exit()

classes = ['bark','cling','comand','eat','handlr','run','victim','shake','sniff','stop','walk']
#draw_heatmap(arra, classes, classes)

