import numpy as np
import torch
import yaml
import os
from torch.utils.data import DataLoader
from torchvision import transforms as T
from learned_cost_map.terrain_utils.terrain_map_tartandrive import TerrainMap, get_local_path
from learned_cost_map.dataloader.TartanDriveDataset import DatasetBase, data_transform
from learned_cost_map.dataloader.TartanDriveBalancedDataset import BalancedTartanDrive, balanced_data_transform
from learned_cost_map.dataloader.WandaDataset import DatasetBaseWanda, wanda_data_transform
from learned_cost_map.dataloader.WandaBalancedDataset import BalancedWandaDataset, balanced_wanda_transform

import time

'''
MultiEpochsDataLoader and _RepeatSampler were taken from here: https://github.com/rwightman/pytorch-image-models/pull/140/files
'''
class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def get_FFM_freqs(data_size, scale=10.0, num_features=16):
    '''Get frequencies to be used for Fourier Feature Mapping (https://arxiv.org/abs/2006.10739) based on dimensionality of data and scale. These frequencies will be fixed during training time.

    Args:
        - data_size:
            Int, K representing the dimensionality of the input data
        - scale:
            Float, scale by which a random variable sampled from a standard Gaussian will be scaled.
    Returns:
        - B:
            Tensor of size (num_features, K). This will be used as [cos(2pi*B*data), sin(2pi*B*data)] in FourierFeatureMapping
    '''
    B = torch.normal(mean=0, std=1, size=(num_features, data_size)) * scale

    return B



def FourierFeatureMapping(data, B):
    '''Performs Fourier Feature Mapping as described in https://arxiv.org/abs/2006.10739

    Args:
        - data:
            Tensor of size (batch, K), where K is the dimensionality of the data
        - B:
            Tensor of size (num_features, K) of Fourier frequencies. This will be used as [cos(2pi*B*data), sin(2pi*B*data)] in FourierFeatureMapping 
    Returns:
        - fourier_data:
            Tensor of size (2*K*num_features,) correspoding to gamma(data) = [cos(2*pi*B*data), sin(2*pi*B*data)] where B is obtained using get_FFM_freqs
    '''
    # import pdb;pdb.set_trace()
    # Reshape data and B so that they can be batch matrix multiplied
    if len(data.shape) > 1:
        data = data.view(*data.shape, 1)
        B = B.view(1, *B.shape)  

    data = data.float()
    B = B.float()
    
    data_cos = torch.cos(2*np.pi*torch.matmul(B,data)).squeeze()
    data_sin = torch.sin(2*np.pi*torch.matmul(B,data)).squeeze()

    fourier_data = torch.cat([data_cos, data_sin], dim=-1)

    return fourier_data

def get_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val, map_config, augment_data=False, use_multi_epochs_loader=True):
    

    print("Inside utils/get_dataloaders")
    before_time = time.time()

    with open(map_config, "r") as file:
        map_info = yaml.safe_load(file)
    map_metadata = map_info["map_metadata"]
    crop_params = map_info["crop_params"]

    datatypes = "imgc,heightmap,rgbmap,odom,cost,patches"
    base_mod_lengths = [1,1,1,10,10,1]
    modality_lengths = [seq_length*l for l in base_mod_lengths]

    train_set = DatasetBase(train_split,
                            map_metadata,
                            crop_params,
                            dataroot= data_root_dir,
                            datatypes = datatypes,
                            modalitylens = modality_lengths,
                            transform=data_transform,
                            imu_freq = 10,
                            frame_skip = 0, 
                            frame_stride=5,
                            augment_data=augment_data)
    val_set = DatasetBase(val_split,
                        map_metadata,
                        crop_params,
                        dataroot= data_root_dir,
                        datatypes = datatypes,
                        modalitylens = modality_lengths,
                        transform=data_transform,
                        imu_freq = 10,
                        frame_skip = 0, 
                        frame_stride=5,
                        augment_data=False)



    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader
    else:
        loader_class = DataLoader

    
    train_loader = loader_class(dataset=train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)

    val_loader = loader_class(dataset=val_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def get_balanced_dataloaders(batch_size, data_root_dir, train_lc_dir, train_hc_dir, val_lc_dir, val_hc_dir, map_config, augment_data=True, high_cost_prob=None, use_multi_epochs_loader=True):
    
    data_train_lc_dir = os.path.join(data_root_dir, train_lc_dir)
    data_train_hc_dir = os.path.join(data_root_dir, train_hc_dir)

    data_val_lc_dir = os.path.join(data_root_dir, val_lc_dir)
    data_val_hc_dir = os.path.join(data_root_dir, val_hc_dir)

    with open(map_config, "r") as file:
        map_info = yaml.safe_load(file)
    map_metadata = map_info["map_metadata"]
    crop_params = map_info["crop_params"]

    train_set = BalancedTartanDrive(data_train_lc_dir, data_train_hc_dir, map_metadata, crop_params, balanced_data_transform, augment_data=augment_data, high_cost_prob=high_cost_prob)

    val_set = BalancedTartanDrive(data_val_lc_dir, data_val_hc_dir, map_metadata, crop_params, balanced_data_transform, augment_data=False, high_cost_prob=high_cost_prob)

    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader
    else:
        loader_class = DataLoader

    train_loader = loader_class(dataset=train_set, batch_size=batch_size, shuffle=False)
    val_loader = loader_class(dataset=val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_wanda_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val, augment_data=False, use_multi_epochs_loader=True, map_metadata=None, crop_params=None):
    
    print("Inside utils/get_wanda_dataloaders")
    before_time = time.time()


    datatypes = "imgc,heightmap,rgbmap,odom,cost,patches"
    base_mod_lengths = [1,1,1,10,10,1]
    modality_lengths = [seq_length*l for l in base_mod_lengths]

    train_set = DatasetBaseWanda(train_split,
                            dataroot= data_root_dir,
                            datatypes = datatypes,
                            modalitylens = modality_lengths,
                            transform=wanda_data_transform,
                            imu_freq = 10,
                            frame_skip = 0, 
                            frame_stride=5,
                            augment_data=augment_data,
                            map_metadata=map_metadata,
                            crop_params=crop_params)

    val_set = DatasetBaseWanda(val_split,
                        dataroot= data_root_dir,
                        datatypes = datatypes,
                        modalitylens = modality_lengths,
                        transform=wanda_data_transform,
                        imu_freq = 10,
                        frame_skip = 0, 
                        frame_stride=5,
                        augment_data=False,
                        map_metadata=map_metadata,
                        crop_params=crop_params)



    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader
    else:
        loader_class = DataLoader

    
    train_loader = loader_class(dataset=train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)

    val_loader = loader_class(dataset=val_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def get_balanced_wanda_dataloaders(batch_size, data_root_dir, train_lc_dir, train_hc_dir, val_lc_dir, val_hc_dir, map_config, augment_data=True, high_cost_prob=None, use_multi_epochs_loader=True):
    
    data_train_lc_dir = os.path.join(data_root_dir, train_lc_dir)
    data_train_hc_dir = os.path.join(data_root_dir, train_hc_dir)

    data_val_lc_dir = os.path.join(data_root_dir, val_lc_dir)
    data_val_hc_dir = os.path.join(data_root_dir, val_hc_dir)

    with open(map_config, "r") as file:
        map_info = yaml.safe_load(file)
    map_metadata = map_info["map_metadata"]
    crop_params = map_info["crop_params"]

    train_set = BalancedWandaDataset(data_train_lc_dir, data_train_hc_dir, map_metadata, crop_params, balanced_data_transform, augment_data=augment_data, high_cost_prob=high_cost_prob)

    val_set = BalancedWandaDataset(data_val_lc_dir, data_val_hc_dir, map_metadata, crop_params, balanced_data_transform, augment_data=False, high_cost_prob=high_cost_prob)

    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader
    else:
        loader_class = DataLoader

    train_loader = loader_class(dataset=train_set, batch_size=batch_size, shuffle=False)
    val_loader = loader_class(dataset=val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def preprocess_data(input_dict, fourier_freqs=None):
    '''Returns input dictionary and labels.

    Args:
        - input_dict:
            Dictionary loaded from the dataloader
        - fourier_freqs:
            Tensor B of frequencies for Fourier feature mapping of input veocities
    Returns:
        - input_data:
            Dictionary with the following fields:
                {
                    'patches',
                    'vels',
                    'fourier_vels'
                }
        - labels:
            A tensor of pseudo ground-truth labels
    '''
    input_data = {}
    input_data["patches"] = input_dict["patches"].view(-1, *input_dict["patches"].shape[-3:]).float().to('cuda')
    
    odom_tensor = input_dict["odom"]
    vels = torch.linalg.norm(odom_tensor[...,7:10], dim=-1).view(-1,1).float().to('cuda') # view(-1,1) refers to -1 batches, 1 dim for velocity, since it is one dimensional
    #Normalize velocity:
    vels=torch.clamp(vels/20.0, min=0.0, max=1.0)
    input_data["vels"] = vels

    if fourier_freqs is not None:
        input_data['fourier_vels'] = FourierFeatureMapping(vels, fourier_freqs.to('cuda')).float().to('cuda')
    else:
        input_data["fourier_vels"] = None


    labels = input_dict["cost"].view(-1).float().to('cuda')
    return input_data, labels

def avg_dict(all_metrics):
    keys = all_metrics[0].keys()
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = np.mean([all_metrics[i][key].cpu().detach().numpy() for i in range(len(all_metrics))])
    return avg_metrics


bgr_to_rgb = lambda img: img[[2,1,0],:,:] 
    
transform_to_img = T.Compose([
    T.Normalize(mean = [0., 0., 0.], std  = [1/0.229, 1/0.224, 1/0.225]),
    T.Normalize(mean = [-0.485, -0.456, -0.406], std  = [1., 1., 1.]),
    T.Lambda(bgr_to_rgb),
    T.ToPILImage(),
    np.asarray
])


def tensor_to_img(img_tensor):
    '''Converts a tensor representing an image into a numpy array that can be directly used for plotting purposes.

    Args:
        img_tensor:
            Tensor(C,H,W)->Float or Tensor(H,W)->Float representing image
    Returns:
        img_plot:
            Array(H,W,C)->Uint8 or Array(H,W)->Uint8 image ready to be plotted
    '''
    if img_tensor.shape[0] == 1 or len(img_tensor.shape) < 3:
        raise NotImplementedError

    img_plot = transform_to_img(img_tensor)

    return img_plot

def tensor_to_heightmap(heightmap_tensor):
    '''Converts a heightmap tensor into an array that can be directly used for plotting purposes.

    Args:
        heightmap_tensor:
            Tensor(C,H,W)->Float representing heightmap, where C=5 and corresponds to the following channels: min height, max height, mean height, std height, invalid mask. These maps are in the [0,1] range,and were normalized using values min=-2, max=2, and x_norm = x-min/(max-min).
        Returns:
            heightmap_array:
                Array(H,W,C)->Float heightmap, where C=2, and in this case correspond to the unnormalized min and max values for the heightmap.
    '''

    unnorm_height_map = 4*heightmap_tensor[:-1,] - 2
    # import pdb;pdb.set_trace()
    # nan_idx = torch.nonzero(heightmap_tensor[-1] == 1)
    # for channel in range(unnorm_height_map.shape[0]):
    #     unnorm_height_map[channel][nan_idx] = torch.nan
    heightmap_array = unnorm_height_map[0:2].permute(1,2,0).detach().cpu().numpy()

    return heightmap_array


def patches_to_imgs(patches_tensor):
    '''Converts a tensor of map patches into two numpy arrays: One that contains the batched RGB data for each patch into a form that can be directly plotted if iterated over, and one that contains the batched heightmap information.

    Args:
        patches_tensor:
            Tensor(N, C, H, W)->Float representing N map patches, where N corresponds to the lookahead.
    Returns:
        rgb_maps:
            Array(N, H, W, C)->Uint8 containing RGB map information for each patch, where C=3
        height_maps:
            Array(N, H, W, C)->Float containing height map information for each patch, where C=2. TBD whether the two channel dimensions correspond to min/max or mean/std.
    '''
    # if patches_tensor.shape[0] == 1 or len(patches_tensor.shape) < 4:
    #     raise NotImplementedError
    if len(patches_tensor.shape) < 4:
        raise NotImplementedError

    # import pdb;pdb.set_trace()
    rgb_maps_tensor = patches_tensor[:,0:3, :, :]
    height_maps_tensor = patches_tensor[:, 3:, :, :]

    # Process rgb maps
    rgb_imgs = []
    for img in rgb_maps_tensor:
        rgb_img = transform_to_img(img)
        rgb_imgs.append(rgb_img)
    rgb_maps = np.stack(rgb_imgs, axis=0)

    # Process height maps
    # Remember: need to unnormalize
    height_maps = []
    for hm in height_maps_tensor:
        height_map = tensor_to_heightmap(hm)
        height_maps.append(height_map)
    height_maps = np.stack(height_maps, axis=0)

    return rgb_maps, height_maps