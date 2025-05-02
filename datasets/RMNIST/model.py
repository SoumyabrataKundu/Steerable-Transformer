import os
import sys
import h5py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import Steerable.nn as snn
from Steerable.datasets.hdf5 import HDF5

class Model(nn.Module):
    def __init__(self, n_radius, max_m, interpolation, restricted) -> None:
        super(Model, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_theta = 40
        restricted = bool(restricted)
        conv_first = restricted

        self.network = nn.Sequential(
            snn.SE2ConvType1(1,  24, 5, n_radius, n_theta, max_m, padding='same', conv_first=conv_first, interpolation_type=interpolation),
            snn.SE2BatchNorm(), 
            snn.SE2CGNonLinearity(max_m),
            snn.SE2BatchNorm(),
            snn.SE2ConvType2(24,  48, 5, n_radius, n_theta, max_m, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),
            snn.SE2BatchNorm(),

            snn.SE2AvgPool(2),

            snn.SE2ConvType2(48, 48, 5, n_radius, n_theta, max_m, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation),
            snn.SE2BatchNorm(),
            snn.SE2CGNonLinearity(max_m),
            snn.SE2BatchNorm(),
            snn.SE2ConvType2(48, 96, 5, n_radius, n_theta, max_m, padding='same', restricted=restricted, conv_first=conv_first, interpolation_type=interpolation), 
            snn.SE2BatchNorm(),

            snn.SE2AvgPool(2),

            snn.SE2ConvType2(96, 64, 7, n_radius, n_theta, max_m, restricted=restricted, conv_first=conv_first, interpolation_type=interpolation), #  1 X  1
            snn.SE2BatchNorm(),

            snn.SE2Pooling(),
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(),
            #torch.nn.Dropout(0.7),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        x = x.type(torch.cfloat)
        return self.network(x)
  






class AddGaussianNoise:
    def __init__(self, sd: float):
        self.sd = sd

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.sd
        return tensor + noise




def get_datasets(data_path, noise=0):
    # Load the dataset
    data_file = h5py.File(os.path.join(data_path, 'RotMNIST.hdf5'), 'r')
    
    # Transformations
    if noise>0:
        image_transform = transforms.Compose([
            AddGaussianNoise(noise),
            transforms.Normalize(mean=0, std = 1)
            ])
    else:
        image_transform = transforms.Compose([
            transforms.Normalize(mean=0, std = 1)
            ])   
 
    # Load datasets
    train_dataset = HDF5(data_file, mode='train', image_transform=image_transform)
    val_dataset = HDF5(data_file, mode='val', image_transform=image_transform)
    test_dataset = HDF5(data_file, mode='test', image_transform=image_transform)

    train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset]) 
    val_dataset = None

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset} 
