import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image
from natsort import natsorted

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=0.1):
        self.batch_size = batch_size
        self.test_percent = test_percent


        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')
        
        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.labels_dir = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()
        

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)
            print(endId)

        while current < endId:

            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!
            data_image = Image.open(self.data_files[current])
            label_image = Image.open(self.labels_dir[current])
            if self.mode == 'train':
                data_image = data_image.resize((256,256))
                label_image = label_image.resize((256,256), resample = Image.NEAREST)
            
           
            data_image = np.asarray(data_image)
            label_image = np.asarray(label_image)
            data_max = data_image.max()
            data_min = data_image.min()
            data_image = (data_image - data_min)/data_max
            rotation_image = self.data_aug(data_image,mode = "r")
            rotation_l_image = self.data_aug(label_image,mode = "r")
            filp_image = self.data_aug(data_image,mode = "f")
            filp_l_image = self.data_aug(label_image,mode = "f")
            gamma_image = self.data_aug(data_image,mode = "g")
            current += 1
            if self.mode == 'train':
                yield (data_image, label_image) 
                yield (rotation_image, rotation_l_image) 
                yield (filp_image, filp_l_image) 
                yield (gamma_image, label_image) 
            if self.mode == 'test':
                yield (data_image, label_image) 


    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))
    
    def data_aug(self, file, mode):
        h = file.shape[0]
        w = file.shape[1]
        h = int(h)
        w = int(w)
        result = np.zeros([h,w])
        if mode == "f":
            for i in range(h+1//2):
                temp = file[i,:]
                result[i-1,:] = file[h-i-1] 
                result[h-i-1,:] = temp
            return result
        if mode == "r":
            for i in range(h+1//2):
                for j in range(w):
                    result[i,j] = file[j-1,i-1]
                    result = result[0:h, 0:w]
            return result
        if mode == "g":
            gamma = 0.9
            inv_gamma = 1/gamma
            result = ((file/255) ** inv_gamma) * 255
            return result



        