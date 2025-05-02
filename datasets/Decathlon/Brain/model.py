import torch
import torchvision


#from SteerableSegmenter2D.conv_layers import *
#from SteerableSegmenter2D.transformer_layers import SE2TransformerEncoder, SE2LinearDecoder

from Steerable.nn import *

class Model(torch.nn.Module):
    def __init__(self, n_radius, max_m) -> None:
        super(Model, self).__init__()
        
        device='cuda' 
        n_theta = 40
        self.num_classes = 4
        encoder_dim = [64,32,16]
        decoder_dim = [64,32,16]

        self.convolution_stem1 = nn.Sequential(
            SE3Conv(4,[8,4,2],7, n_radius, n_theta, stride=2),
            SE3NormNonLinearity([8,4,2]),
            SE3Conv([8,4,2],[16,8,4],7, n_radius, n_theta, padding='same'),
            SE3BatchNorm(),
        )
        
        self.pool1 = SE3AvgPool(8)
  
        self.convolution_stem2 =  nn.Sequential(
            SE3Conv([16,8,4],[32,16,8],5, n_radius, n_theta, padding='same'),
            SE3CGNonLinearity([32,16,8]),
            SE3Conv([32,16,8],encoder_dim,5, n_radius, n_theta, padding='same'),
            SE3BatchNorm(),
        )

        self.pool2 = SE3AvgPool(4)

        self.encoder = SE3TransformerEncoder(encoder_dim, 4, n_layers = 2, add_pos_enc=False)
        #self.decoder = SE3LinearDecoder(encoder_dim, decoder_dim)
        #self.decoder = SE3TransformerDecoder(encoder_dim, 4, self.num_classes, n_layers=2, add_pos_enc=True)
        
        self.convolution_head1 = nn.Sequential(
            SE3Conv(decoder_dim,[32,16,8],5, n_radius, n_theta, padding = 'same'),
            SE3NormNonLinearity([32,16,8]),
            SE3Conv([32,16,8],[16,8,4],5, n_radius, n_theta, padding = 'same'),
            SE3BatchNorm(),
        )

        self.convolution_head2 = nn.Sequential(
            SE3Conv([16,8,4],[8,4,2],7, n_radius, n_theta, padding = 'same'),
            SE3NormNonLinearity([8,4,2]),
            SE3BatchNorm(),
	    SE3Conv([8,4,2],self.num_classes,7, n_radius, n_theta, padding = 'same'),
        )
       
        #self.embed = SE3ClassEmbedings(encoder_dim, [8,4,2])
        
    def forward(self, x):
        x_shape = x.shape
        x = x.type(torch.cfloat)
        
        # Downsampling
        stem1 = self.convolution_stem1(x)
        x = self.pool1(stem1)
        stem2 = self.convolution_stem2(x)
        x = self.pool2(stem2)

        # Encoder
        x = self.encoder(x)

        # Decoder
        #x,classes = self.decoder(x)

        # Upsampling
        x, channels = merge_channel_dim(x)
        x = nn.functional.interpolate(x.real, size=stem2[0].shape[-3:], mode="trilinear") + \
                  1j * nn.functional.interpolate(x.imag, size=stem2[0].shape[-3:], mode="trilinear")
        x = split_channel_dim(x, channels=channels)
        x = [x[l] + stem2[l] for l in range(len(x))] # skip connection
        x = self.convolution_head1(x)

        x, channels = merge_channel_dim(x)
        x = nn.functional.interpolate(x.real, size=stem1[0].shape[-3:], mode="trilinear") + \
                  1j * nn.functional.interpolate(x.imag, size=stem1[0].shape[-3:], mode="trilinear")
        x = split_channel_dim(x, channels=channels)
        x = [x[l] + stem1[l] for l in range(len(x))] # skip connection
        x = self.convolution_head2(x)

        x = x[0].squeeze(1)
        x = nn.functional.interpolate(x.real, size=x_shape[-3:], mode="trilinear") + \
                  1j * nn.functional.interpolate(x.imag, size=x_shape[-3:], mode="trilinear")
        #x = split_channel_dim(x, channels=channels) 
        #x = self.embed(x, classes)
        return x.abs()

#######################################################################################################################
###################################################### Dataset ########################################################
####################################################################################################################### 



import torch
import os
import torchvision.transforms as transforms
import h5py

class InterpolateToSize:
    def __init__(self, size, mode='nearest-exact'):
        self.size = size
        self.mode = mode

    def __call__(self, image):

        if image.ndim - len(self.size)>=0 and image.ndim - len(self.size) <= 2:
            resized_image = torch.nn.functional.interpolate(image.reshape(*[1]*(2 - image.ndim + len(self.size)), *image.shape).float(), size=self.size ,mode=self.mode)
            resized_image = resized_image.reshape(*image.shape[:-3], *self.size)
        else:
            raise IndexError("")

        return resized_image

class Decathlon(torch.utils.data.Dataset):
    def __init__(self, file, image_transform = None, target_transform = None) -> None:

        self.mode = 'train'
        self.file = file
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.n_samples = len(self.file[self.mode+'_targets'])

    def __getitem__(self, index):
        # Reading from file
        input = torch.from_numpy(self.file[self.mode + '_inputs'][index]).float()
        target = torch.tensor(self.file[self.mode + '_targets'][index]).long()

        # Applying trasnformations
        if self.image_transform is not None:
            input = self.image_transform(input)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target.long()

    def __len__(self):
        return len(self.file[self.mode+'_targets'])


def get_datasets(data_path):
    # Load the dataset
    data_file = h5py.File(os.path.join(data_path, 'Brain.hdf5'), 'r')

    # Transformations
    image_transform = transforms.Compose([
        #InterpolateToSize((120,120,75), mode='trilinear'),
        transforms.Normalize(mean=0, std = 1)
        ])

    target_transform = transforms.Compose([
        InterpolateToSize((120,120,75), mode='nearest-exact'),
        ])

    # Load datasets
    datasets = Decathlon(data_file, image_transform=image_transform, target_transform=None)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(datasets, [0.5, 0.2, 0.3])

    return {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}
