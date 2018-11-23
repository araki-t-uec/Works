#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-


import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
#import model
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0" #,2,3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize((299,299)),
#    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
preprocess2 = transforms.Compose([
#    transforms.Resize((224,224)),
#    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)



model = torchvision.models.inception_v3(pretrained=True)
print(model)
model = model.to(device)
model.eval()
#model.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

clas_num = 398
save_dir = "./Img/Dream"
tmp_name = "0.jpg"
tmp_file = os.path.join(save_dir, tmp_name)
#image = Image.open("./data/ImageNet/train/n11793779/n11793779_19911.JPEG")
#image = Image.open("./Img/ege.jpg")
#image = Image.open("./data/ImageNet/train/n12755727/n12755727_1312.JPEG")
image = Image.open("data/ImageNet/train/n01693334/n01693334_2518.JPEG")
# img_tensor = preprocess(image) # torch.Size([3, 224, 224])
# # print(img_tensor.shape)
# img_tensor.unsqueeze_(0)       # torch.Size([1, 3, 224, 224])
# # print(img_tensor.size())
# input_tensor = torch.autograd.Variable(img_tensor)
# input_tensor = input_tensor.to(device)
# input_tensor.requires_grad=True
# torchvision.utils.save_image(denorm(input_tensor.data.cpu()), "./Img/ege.jpg")

for i in range(20000):
    img_tensor = preprocess(image) # torch.Size([3, 224, 224])
    img_tensor.unsqueeze_(0)       # torch.Size([1, 3, 224, 224])
    input_tensor = torch.autograd.Variable(img_tensor)
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad=True
    exit()
#    torchvision.utils.save_image(denorm(input_tensor.data.cpu()), "./Img/ege.jpg")
#    print(i)
#    continue



    output = model(input_tensor)
    # print(np.argmax(output.data.cpu().numpy()))
    # print(out.topk(5))
    clas = [clas_num]
    clas = torch.tensor(clas).to(device)
    loss = criterion(output, clas)
    loss.backward()
    #print(vgg16.features[0].weight.grad) #.data.norm())
    #print(type(vgg16.features[0].weight.grad)) #.data.norm())
    
    # img_tensor = preprocess(image)
    # img_tensor.unsqueeze_(0)       # torch.Size([1, 3, 224, 224])
    # output_tensor = torch.autograd.Variable(img_tensor)
    # output_image = output_tensor.to(device)
    #print(input_tensor)
    #print(input_tensor.grad)
    #print(input_tensor.requires_grad)
    #print(img_tensor)
    #print(torch.max(img_tensor - org))


    #torchvision.utils.save_image(denorm(input_tensor.data.cpu()), "./Img/ege.jpg")
    #print(i)
    #continue

    
    print(i)
    #output_image = input_tensor-input_tensor.grads
    output_image = input_tensor-input_tensor.grad*10
    #input_tensor = input_tensor-input_tensor.grad
    #print(output_image.size())

    if i%100 == 0:
        #save_name = str(i) +".jpg"
        save_name = str(i).zfill(5) +".jpg"
        save_file = os.path.join(save_dir, save_name)
        torchvision.utils.save_image(denorm(output_image.data.cpu()), save_file)
        print("save")
        print(np.argmax(output.data.cpu().numpy()))
    if i%400 == 399:
        clas_num += 10
        print("class num changed to :", clas_num)
    torchvision.utils.save_image(denorm(output_image.data.cpu()), tmp_file)
    image = Image.open(tmp_file)
