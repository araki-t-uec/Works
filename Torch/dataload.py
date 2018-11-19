import torch
import torchvision.transforms as transforms
import os
from PIL import Image


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
        image = Image.open(img_path)
        # #画像へ処理を加える
        t = transforms.Resize((224, 224))
        image = t(image)
        if self.transform:
            image = self.transform(image)

        return image, int(label)-1


class YourDataset(torch.utils.data.Dataset):
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
        # image = Image.open(img_path)
        image = Image.open(filename)
        image = image.convert('RGB')
        # #画像へ処理を加える
        t = transforms.Resize((224, 224))
        image = t(image)
        if self.transform:
            image = self.transform(image)

        return image, label 
