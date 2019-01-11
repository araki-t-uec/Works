import torch
import torchvision.transforms as transforms
import os, re
from PIL import Image
import numpy as np
import librosa


class MulchVideoset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, data_dir, classes, transform=None):
        self.image_dataframe = []
        self.transform = transform
        
        f = open(annotation_file)
#        classes = {"bark":0,  "cling":1, "command":2, "eat-drink":3, "look_at_handler":4, "run":5, "see_victim":6, "shake":7, "sniff":8, "stop":9, "walk-trot":10}
        for aline in f:
            match = re.search(r'\d+ \d+_\d+.jpg .*', aline)
            video = match.group(0).split(" ")[0]
            frame = match.group(0).split(" ")[1]
            ml_class =  match.group(0).split(" ")[2:]
            label = [0]*len(classes)
            for aclass in ml_class:
                try:
                    label[classes[aclass]] = 1
                except:
                    pass
            ##
            #print(os.path.join(data_dir, video, frame), label)
            self.image_dataframe.append([os.path.join(data_dir, video, frame), label])

            
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        filename = self.image_dataframe[idx][0]
        label = self.image_dataframe[idx][1]
        img_path = os.path.join(filename)
        # #画像の読み込み
        #img_path = "data/UECFOOD100/1/11292.jpg"
        #label = 1
        image = Image.open(img_path)
        # #画像へ処理を加える
        #t = transforms.Resize((32, 32))
        #image = t(image)
        if self.transform:
            image = self.transform(image)

        return image, np.array(label)

    
class Oneovern(torch.utils.data.Dataset):
    def __init__(self, annotation_file, data_dir, classes, transform=None, n=3):
        ##
        ## For load 1/n frame
        ##
        self.image_dataframe = []
        self.transform = transform
        
        f = open(annotation_file)
        ## classes = {"bark":0,  "cling":1, "command":2,
        ##            "eat-drink":3, "look_at_handler":4,
        ##            "run":5, "see_victim":6, "shake":7,
        ##            "sniff":8, "stop":9, "walk-trot":10}
        i = 0
        for aline in f:
            i += 1
            if i % n == 0:
                match = re.search(r'\d+ \d+_\d+.jpg .*', aline)
                video = match.group(0).split(" ")[0]
                frame = match.group(0).split(" ")[1]
                ml_class =  match.group(0).split(" ")[2:]
                label = [0]*len(classes)
                for aclass in ml_class:
                    try:
                        label[classes[aclass]] = 1
                    except:
                        pass
                ##
                #print(os.path.join(data_dir, video, frame), label)
                self.image_dataframe.append([os.path.join(data_dir, video, frame), label])

            
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        filename = self.image_dataframe[idx][0]
        label = self.image_dataframe[idx][1]
        img_path = os.path.join(filename)
        # #画像の読み込み
        #img_path = "data/UECFOOD100/1/11292.jpg"
        #label = 1
        image = Image.open(img_path)
        # #画像へ処理を加える
        #t = transforms.Resize((32, 32))
        #image = t(image)
        if self.transform:
            image = self.transform(image)

        return image, np.array(label)




class Stillandmulchoptic(torch.utils.data.Dataset):
    def __init__(self, annotation_file, still_dir, optic_dir, classes, transform=None, n=5):
        ##
        ## For load 1/n still frame  | |{|}| | | |{|}| | | |
        ##          n-1 optic frame  {| | | |}|{| | | |}| |
        ##
        self.image_dataframe = []
        self.transform = transform
        self.n = n
        self.still_dir = still_dir
        self.optic_dir = optic_dir
        f = open(annotation_file)
        ## classes = {"bark":0,  "cling":1, "command":2,
        ##            "eat-drink":3, "look_at_handler":4,
        ##            "run":5, "see_victim":6, "shake":7,
        ##            "sniff":8, "stop":9, "walk-trot":10}
        i = 0
        self.sur = int(n/2) 
        for aline in f:
            i += 1
            if i % n == self.sur+1:
                match = re.search(r'\d+ \d+_\d+.jpg .*', aline)
                video = match.group(0).split(" ")[0]
                frame = match.group(0).split(" ")[1]
                ml_class =  match.group(0).split(" ")[2:]
                label = [0]*len(classes)
                for aclass in ml_class:
                    try:
                        label[classes[aclass]] = 1
                    except:
                        pass
                ##
                #print(os.path.join(data_dir, video, frame), label)
                self.image_dataframe.append([video, frame, label])

            
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        videoname = self.image_dataframe[idx][0] ## 20150801
        framename = self.image_dataframe[idx][1] ## 20150801_000003.jpg
        label = self.image_dataframe[idx][2]
        img_path = os.path.join(self.still_dir, videoname, framename)
        
        # #画像の読み込み
        images = []
        image = Image.open(img_path)
        images.append(image)
        
        number = int(framename.split("_")[-1].split(".")[0]) ## 000003 -> 3
        dist = self.n - self.sur*3 ## if n=5, dist=-1
        for i in range(self.n-1):
            number += dist
            filename = videoname+"_"+"%06d"%number+".jpg" ## 20150801_000002.jpg
            img_path = os.path.join(self.optic_dir, videoname, filename) ## opt/0801/0801_002.jpg
            try:
                image = Image.open(img_path)
                images.append(image)
                dist += 1
            except:
                pass
            
        # #画像へ処理を加える
        transformed_images = []
        if self.transform:
            for i in images:
                image = self.transform(i)
                transformed_images.append(image)
        return transformed_images, np.array(label)

    

class Stlandopt(torch.utils.data.Dataset):
    def __init__(self, annotation_file, still_dir, optic_dir, classes, transform=None, n=5):
        self.image_dataframe = []
        self.transform = transform
        self.n = n
        self.still_dir = still_dir
        self.optic_dir = optic_dir
        f = open(annotation_file)
#        i = 0
#        self.sur = int(n/2) 
        for aline in f:
#            i += 1
#            if i % n == self.sur+1:
            match = re.search(r'\d+ \d+_\d+.jpg .*', aline)
            video = match.group(0).split(" ")[0]
            frame = match.group(0).split(" ")[1]
            ml_class =  match.group(0).split(" ")[2:]
            label = [0]*len(classes)
            for aclass in ml_class:
                try:
                    label[classes[aclass]] = 1
                except:
                    pass
            ##
            #print(os.path.join(data_dir, video, frame), label)
            self.image_dataframe.append([video, frame, label])

            
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        videoname = self.image_dataframe[idx][0] ## 20150801
        framename = self.image_dataframe[idx][1] ## 20150801_000001.jpg
        label = self.image_dataframe[idx][2]

        opt_path = os.path.join(self.still_dir, videoname, framename)
        stl_path = os.path.join(self.optic_dir, videoname, framename)
        
        # #画像の読み込み
        images = []
        opt = Image.open(opt_path)
        stl = Image.open(stl_path)
        images.append(stl)
        images.append(opt)
            
        # #画像へ処理を加える
        transformed_images = []
        if self.transform:
            for i in images:
                image = self.transform(i)
                transformed_images.append(image)
        return transformed_images, np.array(label)

    
class DogSounds(torch.utils.data.Dataset):
    def __init__(self, annotation_file, sound_dir, classes, transform=None):
        self.image_dataframe = []
        self.transform = transform
        self.sound_dir = sound_dir
        f = open(annotation_file)
        for aline in f:
            match = re.search(r'\d+ \d+_\d+.wav .*', aline)
            video = match.group(0).split(" ")[0]
            frame = match.group(0).split(" ")[1]
            ml_class =  match.group(0).split(" ")[2:]
            label = [0]*len(classes)
            for aclass in ml_class:
                try:
                    label[classes[aclass]] = 1
                except:
                    pass
            ##
            #print(os.path.join(sound_dir, video, frame), label)
            self.image_dataframe.append([video, frame, label])

            
    def __len__(self):
        return len(self.image_dataframe)

    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        audio_name = self.image_dataframe[idx][0] ## 20150801
        frame_name = self.image_dataframe[idx][1] ## 20150801_000016.wav
        label = self.image_dataframe[idx][2]
        wav_path = os.path.join(self.sound_dir, audio_name, frame_name) ## Sound/2015/0016.wav
        
        ## wav-file の読み込み
        x, fs = librosa.load(wav_path, sr=48000)
        mfccs = librosa.feature.mfcc(x, sr=fs) ##  (20, 92)
        sound = torch.Tensor(mfccs).unsqueeze(0) ##  (1, 20, 92) 
        return sound, np.array(label)

    
