import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dataframe = []
        self.data_dir = data_dir
        class_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        for i in class_dirs:
            i_files = os.listdir(os.path.join(data_dir, i))
            for j in i_files:
                self.image_dataframe.append([j,i])

        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transform

        
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        filename = self.image_dataframe[idx][0]
        label = self.image_dataframe[idx][1]
        img_path = os.path.join(self.data_dir, label, filename)
        # #画像の読み込み
        # img_path = "./data/ImageNet/train/n01644900/n01644900_9597.JPEG"
        image = Image.open(img_path)
        # #画像へ処理を加える
        if self.transform:
            image = self.transform(image)

        return image, int(label)-1


class YourDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        ignorelist=["n02487347_1956.JPEG",".n06595351_20417.JPEG.lVUKt8"]
        self.image_dataframe = []
        self.data_dir = data_dir
        class_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        count = 0
        self.categories = {}
        for i in class_dirs:
            i_files = os.listdir(os.path.join(data_dir, i))
            if i not in self.categories.keys():
                self.categories[i] = count
                count += 1
            for j in i_files:
                if j not in ignorelist:
                    img_path = os.path.join(data_dir, i, j)
                    self.image_dataframe.append([img_path, self.categories[i]])
                else:
                    print("ignore :", os.path.join(data_dir, i, j))
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transform
        
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        filename = self.image_dataframe[idx][0]
        label = self.image_dataframe[idx][1]
        #img_path = os.path.join(self.data_dir, label, filename)
        # #画像の読み込み
        #filename = "./data/ImageNet/train/n02487347/n02487347_1956.JPEG"
        #filename = "./data/ImageNet/train/n01669191/n01669191_10054.JPEG"

        print(filename)
        image = Image.open(filename)
        image = image.convert('RGB')
        # #画像へ処理を加える
        if self.transform:
            image = self.transform(image)

        return image, label 


class AVIload(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dataframe = []
        self.data_dir = data_dir
        class_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        count = 0
        self.categories = {}
        for i in class_dirs:
            i_files = os.listdir(os.path.join(data_dir, i))
            if i not in self.categories.keys():
                self.categories[i] = count
                count += 1
            for j in i_files:
                img_path = os.path.join(data_dir, i, j)
                print(img_path)
                self.image_dataframe.append([img_path, self.categories[i]])
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transform
        
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        filename = self.image_dataframe[idx][0]
        label = self.image_dataframe[idx][1]
        #img_path = os.path.join(self.data_dir, label, filename)
        # #画像の読み込み
        print(filename)
        #filename("./data/ImageNet/train/n11906917/n11906917_2283.JPEG")
        image = Image.open(filename)
        image = image.convert('RGB')
        # #画像へ処理を加える
        t = transforms.Resize((224, 224))
        image = t(image)
        if self.transform:
            image = self.transform(image)

        return image, label 

class Cifarset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dataframe = []
        self.data_dir = data_dir
        class_dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        for i in class_dirs:
            i_files = os.listdir(os.path.join(data_dir, i))
            for j in i_files:
                self.image_dataframe.append([j,i])
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transform

        
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        filename = self.image_dataframe[idx][0]
        label = self.image_dataframe[idx][1]
        img_path = os.path.join(self.data_dir, label, filename)
        # #画像の読み込み
        #img_path = "data/UECFOOD100/1/11292.jpg"
        #label = 1
        image = Image.open(img_path)
        # #画像へ処理を加える
        #t = transforms.Resize((32, 32))
        #image = t(image)
        if self.transform:
            image = self.transform(image)

        return image, np.array(int(label)-1)
