import sys
from SteerableTransformer3D.transformer_layers import *
from SteerableTransformer3D.conv_layers import *


############################################################################################################################
###################################################### Model ###############################################################
############################################################################################################################

class Model(nn.Module):
    def __init__(self, n_radius, maxl) -> None:
        super(Model, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_angle = 40

        self.network = nn.Sequential(
            FintConv3DType1(1, 9, 5, n_radius, n_angle, maxl, padding='same', device = device),     # 32 X 32 X 32
            CGNonLinearity3D(9, 9, maxl),
            FintConv3DType2(9, 18, 4, n_radius, n_angle, maxl, stride = 4, device = device),        # 8 X 8 X 8
            SteerableBatchNorm3D(),


            SE3TransformerEncoder(18, 3, maxl, n_layers = 4, device = device),


            FintConv3DType2(18, 36, 3, n_radius, n_angle, maxl, padding='same', device = device),   #  8 X  8 X  8
            CGNonLinearity3D(36, 36, maxl),
            FintConv3DType2(36, 72, 2, n_radius, n_angle, maxl, stride = 2, device = device),       #  4 X  4 X  4
            SteerableBatchNorm3D(),

            SE3TransformerEncoder(72, 9, maxl, n_layers = 4, device = device),

            FintConv3DType2(72, 144, 4, n_radius, n_angle, maxl, padding='same', device = device), #  4 X  4 X  4

            NormFlatten(),
            #torch.nn.Linear(144, 144),
            #torch.nn.ReLU(),
            torch.nn.Linear(144, 10)
        )

    def forward(self, x):
        return self.network(x)


##############################################################################################################################
################################################# ModelNet10 Dataset ############################################################
##############################################################################################################################

import torch
import torch.utils.data
import h5py
import os

class ModelNet10(torch.utils.data.Dataset):
    def __init__(self, file, mode = 'train', image_transform = None, target_transform = None) -> None:

        if not mode in ["train", "test"]:
            raise ValueError("Invalid mode")

        self.mode = mode
        self.file = file
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.n_samples = len(self.file[mode+'_targets'])

    def __getitem__(self, index):

        # Reading from file
        img = self.file[self.mode + '_images'][index]
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
    data_file = h5py.File(os.path.join(data_path, 'modelnet10_z_transformed.hdf5'), 'r')

    # Transformations
    def image_transform(x):
        x = (torch.from_numpy(x).unsqueeze(0) * 6) -1
        return x

    # Load datasets
    train_dataset = ModelNet10(data_file, mode='train', image_transform=image_transform, target_transform = None)
    test_dataset = ModelNet10(data_file, mode='test', image_transform=image_transform, target_transform = None)

    return {'train' : train_dataset, 'val' : None, 'test' : test_dataset}



def lr_scheduler(epoch, init_lr, decay_epoch, decay_factor):
    """
    Decay initial learning rate in steps at given points
    The learning rate is decreased by decay_factor[i] at decay_epochs[i]
    :param optimizer: the optimizer inheriting from torch.optim.Optimizer
    :param epoch: the current epoch
    :param init_lr: initial learning rate before decaying
    :param decay_epochs: list of epochs when the learning rate should be decayed
    :param decay_factors: float or list of factors by which the learning rate should be decayed at decay_epochs.
                                              when int, same factor at each decay step
                                              factors should be larger than 1
    """
    assert type(decay_epoch) == int
    assert type(decay_factor) == int
    
    lr = init_lr / (decay_factor ** (epoch // decay_epoch))
    
    return lr
