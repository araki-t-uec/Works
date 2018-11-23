#!/usr/local/anaconda3/bin/python3.6  
# -*- coding:utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import torchvision.transforms.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import torch.backends.cudnn as cudnn
import os
from PIL import Image
import dataload
import cv2, numpy


save_dir = "./Img" 
os.environ["CUDA_VISIBLE_DEVICES"]="1" #,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg16 = torchvision.models.vgg16(pretrained=True)

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


batch = 1
epochs = 1
works = 1
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True, num_workers=works)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=True, num_workers=works)


#print(vgg16)
#print(vgg16.state_dict().keys())
#print(vgg16.state_dict()['features.28.weight'])
#y = vgg16.features(x)
#print(y)

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(torchvision.models.vgg16(pretrained = True).features)[:23]
        self.features = torch.nn.ModuleList(features).eval() 
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
#            if ii in {3,8,15,22}:
            if ii in {3,22}:
                results.append(x)
        return results
model = Vgg16()
print(model)


#print(model.forward(image))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
model = model.eval()


from sklearn.metrics import classification_report
pred = []
Y = []
savedir = "./Img"
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor()
    ])

for i, (x,y) in enumerate(test_loader):
    with torch.no_grad():
        output = model(x)

    pred += [int(l.argmax()) for l in output]
    Y += [int(l) for l in y]
    if i%20 == 19:
        org = x[0]
        org = (org.data + 1) / 2
        org = org.clamp_(0, 1)
        org = make_grid(org, nrow=1, padding=0,
                         normalize=False, range=None, scale_each=False)
        org = org.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        #org = org.transpose(2,0,1)
        org = Image.fromarray(org)
        org.save(os.path.join(savedir, str(i)+".png"))
        for image in output:
            print(image.shape)
            #image = image.numpy()
            out = (image.data + 1) / 2
            img = out.clamp_(0, 1)
            grid = make_grid(img, nrow=1, padding=0,
                             normalize=False, range=None, scale_each=False)
            ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            ndarr = ndarr.transpose(2,0,1)

            #for o in range(ndarr.shape[1]-2):
            for o in range(ndarr.shape[0]):
                out = numpy.array([ndarr[o],ndarr[o],ndarr[o]])
                out = out.transpose(1,2,0)
                #print(out.shape)
                #img = Image.fromarray(ndarr)
                #img.save("hogeeeee.jpg")
            #image = F.to_pil_image(image)
            #print(transform(image))
            #print(type(transform(image)))
            size_grid = image.shape[1]
            size_fig = image.shape[2]
            size_grid = 10
            size_fig = 10
            
            print(size_grid)
            print(size_fig)
            fig, ax = plt.subplots(size_grid, size_grid, figsize=(size_fig, size_fig))

            for k, l in itertools.product(range(size_grid), range(size_grid)):
                ax[k, l].get_xaxis().set_visible(False)
                ax[k, l].get_yaxis().set_visible(False)
            
            for m in range(10*10):
                k = m // 10
                l = m % 10
                ax[k, l].cla()


            

            out = out.transpose(1,2,0)
            ax[k,l].imshow(transform(out))
            label = y
            fig.text(0.5, 0.04, label, ha='center')
            name = str(i) + str(size_grid)+".png"
            filename = os.path.join(savedir, name)
            print(type(out))
            out = out.transpose(2,0,1)
            print(out.shape)
            out = Image.fromarray(out)
            out.save(filename)

            #plt.savefig(filename)
            print("seve to:", filename)

            
print(classification_report(Y, pred))
