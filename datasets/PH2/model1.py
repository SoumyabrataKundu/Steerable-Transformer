import os
import h5py

import torch

from Steerable.utils import HDF5, RandomRotate

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.num_classes = 2
        encoder_dim = 104
        decoder_dim = 104

        self.convolution_stem1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,5, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,32,5, padding='same'),
            torch.nn.BatchNorm2d(32),
        )
        
        self.pool1 = torch.nn.AvgPool2d(4)
  
        self.convolution_stem2 =  torch.nn.Sequential(
            torch.nn.Conv2d(32,64,5, padding='same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,encoder_dim,5, padding='same'),
            torch.nn.BatchNorm2d(encoder_dim),

        )

        self.pool2 = torch.nn.AvgPool2d(4)

        self.encoder_decoder = torch.nn.Sequential(
            torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=8, batch_first=True), num_layers=4),
        )
 
        self.convolution_head1 = torch.nn.Sequential(
            torch.nn.Conv2d(2*decoder_dim,64,5, padding = 'same'),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,32,5, padding = 'same'),
            torch.nn.BatchNorm2d(32),
        )

        self.convolution_head2 = torch.nn.Sequential(
            torch.nn.Conv2d(2*32,16,5, padding = 'same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16,self.num_classes,5, padding = 'same'),
        )

        
    def forward(self, x):
        # Downsampling
        ## Downsampling I
        stem1 = self.convolution_stem1(x)
        x = self.pool1(stem1)
        ## Downsampling II
        stem2 = self.convolution_stem2(x)
        x = self.pool2(stem2)

        # Encoder-Decoder
        x_shape = x.shape
        x = self.encoder_decoder(x.flatten(2).transpose(1,2))
        x = x.transpose(1,2).reshape(*x_shape)

        # Upsampling
        ## Upsampling I
        x = torch.nn.functional.interpolate(x, size=stem2.shape[-2:], mode="bilinear")
        x = self.convolution_head1(torch.cat([x, stem2], dim=1)) # skip connection
        ## Upsampling II
        x = torch.nn.functional.interpolate(x, size=stem1.shape[-2:], mode="bilinear")
        x = self.convolution_head2(torch.cat([x, stem1], dim=1)) # skip connection
 
        return x

#######################################################################################################################
###################################################### Dataset ########################################################
#######################################################################################################################

def get_datasets(data_path, rotate=True):
    data_file = h5py.File(os.path.join(data_path, 'PH2_patched256_128.hdf5'), 'r')

    train_dataset = HDF5(data_file, mode='train')
    if rotate:
        train_dataset = RandomRotate(train_dataset)
    val_dataset = HDF5(data_file, mode='val')
    test_dataset = HDF5(data_file, mode='test')

    data_file = h5py.File(os.path.join(data_path, 'PH2.hdf5'))
    eval_val_dataset = HDF5(data_file, mode='val')
    eval_test_dataset = HDF5(data_file, mode='test')

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset, 'eval_val' : eval_val_dataset, 'eval_test' : eval_test_dataset}
