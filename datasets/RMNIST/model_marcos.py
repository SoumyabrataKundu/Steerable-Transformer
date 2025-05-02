from __future__ import division, print_function
import torch
import  torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from torch.autograd import Variable
import random
import sys

sys.path.append("/project2/risi/soumyabratakundu/se_eq_transformer/2D/src/")
from Marcos.layers import *
from Marcos.utils import getGrid

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.main = nn.Sequential(

            RotConv(1, 6, [9, 9], 1, 9 // 2, n_angles=17, mode=1),
            VectorMaxPool(2),
            VectorBatchNorm(6),

            RotConv(6, 16, [9, 9], 1, 9 // 2, n_angles=17, mode=2),
            VectorMaxPool(2),
            VectorBatchNorm(16),

            RotConv(16, 32, [9, 9], 1, 1, n_angles=17, mode=2),
            Vector2Magnitude(),

            nn.Conv2d(32, 128, 1),  # FC1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.7),
            nn.Conv2d(128, 10, 1),  # FC2

        )

    def forward(self, x):
        x = self.main(x.type(torch.float))
        x = x.view(x.size()[0], x.size()[1])

        return x



import torch
import os
import torchvision.transforms as transforms
import h5py

class RotMNIST(torch.utils.data.Dataset):
    def __init__(self, file, mode = 'train', image_transform = None, target_transform = None) -> None:
        
        if not mode in ["train", "test", "val"]:
            raise ValueError("Invalid mode")
        
        self.mode = mode
        self.file = file
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.n_samples = len(self.file[mode+'_targets'])

    def __getitem__(self, index):
        
        # Reading from file
        img = torch.from_numpy(self.file[self.mode + '_images'][index]).unsqueeze(0)
        target = self.file[self.mode + '_targets'][index]
        
        # Applying trasnformations
        if self.image_transform is not None:
            img = self.image_transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return self.n_samples


def get_datasets(data_path):
    # Load the dataset
    data_file = h5py.File(os.path.join(data_path, 'rotated_mnist.hdf5'), 'r')
    
    # Transformations
    image_transform = transforms.Compose([
        transforms.Normalize(mean=0, std = 1) 
        ])
    
    # Load datasets
    train_dataset = RotMNIST(data_file, mode='train', image_transform=image_transform)
    val_dataset = RotMNIST(data_file, mode='val', image_transform=image_transform)
    test_dataset = RotMNIST(data_file, mode='test', image_transform=image_transform)
    
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    test_set_size = int(len(test_dataset) * 0.95)
    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [test_set_size, len(test_dataset) - test_set_size])
    
    transformations = transforms.Compose([
        #transforms.RandomRotation(degrees=(0, 360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std = 1)
        ])
    #test_dataset = torchvision.datasets.MNIST(data_path, train=True, transform=transformations)

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset} 
