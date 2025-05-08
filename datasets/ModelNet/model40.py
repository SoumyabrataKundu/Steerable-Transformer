import sys

sys.path.append('/project2/risi/soumyabratakundu/se_eq_nn/3D/src/')
from SteerableNet3D.layers import *

############################################################################################################################
###################################################### Model ###############################################################
############################################################################################################################

class Model(nn.Module):
    def __init__(self, n_radius, maxl) -> None:
        super(Model, self).__init__()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_theta = 40
        n_phi = 40

        self.network = nn.Sequential(
            FintConv3DType1(1, 9, 5, n_radius, n_theta, n_phi, maxl, padding='same', device = device),     # 32 X 32 X 32
            CGNonLinearity3D(9, 9, maxl),
            FintConv3DType2(9, 18, 3, n_radius, n_theta, n_phi, maxl, padding='same', device = device),    # 32 X 32 X 32

            SteerableBatchNorm3D(),

            FintAvgPool3D(2),                                                                            
            FintConv3DType2(18, 18, 3, n_radius, n_theta, n_phi, maxl, padding='same', device = device),   # 16 X 16 X 16
            CGNonLinearity3D(18, 18, maxl),
            FintConv3DType2(18, 36, 3, n_radius, n_theta, n_phi, maxl, padding='same', device = device),   # 16 X 16 X 16

            SteerableBatchNorm3D(),

            FintAvgPool3D(2),

            FintConv3DType2(36, 36, 3, n_radius, n_theta, n_phi, maxl, padding='same', device = device),   #  8 X  8 X  8
            CGNonLinearity3D(36, 36, maxl),
            FintConv3DType2(36, 72, 3, n_radius, n_theta, n_phi, maxl, padding='same', device = device),   #  8 X  8 X  8

            FintAvgPool3D(2),

            FintConv3DType2(72, 72, 3, n_radius, n_theta, n_phi, maxl, padding='same', device = device),   #  4 X  4 X  4
            CGNonLinearity3D(72, 72, maxl),
            FintConv3DType2(72, 144, 3, n_radius, n_theta, n_phi, maxl, padding='same', device = device),  #  4 X  4 X  4

            SteerableBatchNorm3D(),

            FintConv3DType2(144, 144, 3, n_radius, n_theta, n_phi, maxl, padding='same', device = device), #  4 X  4 X  4

            NormFlatten(),
          torch.nn.Linear(144, 40),
        )

    def forward(self, x):
        return self.network(x)


##############################################################################################################################
################################################# ModelNet10 Dataset ############################################################
##############################################################################################################################

import torch
import torch.utils.data
import h5py


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
    data_file = h5py.File(os.path.join(data_path, 'modelnet40_transformed.hdf5'), 'r')

    # Transformations
    def image_transform(x):
        x = (torch.from_numpy(x).unsqueeze(0) * 6) -1
        return x

    # Load datasets
    train_dataset = ModelNet10(data_file, mode='train', image_transform=image_transform, target_transform = None)
    test_dataset = ModelNet10(data_file, mode='test', image_transform=image_transform, target_transform = None)

    return {'train' : train_dataset, 'val' : None, 'test' : test_dataset}
