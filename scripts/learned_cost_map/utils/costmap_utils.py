from collections import OrderedDict
import numpy as np
import os
import torch
from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim
from learned_cost_map.trainer.model import CostModel
from learned_cost_map.terrain_utils.terrain_map_tartandrive import TerrainMap
from learned_cost_map.trainer.utils import get_dataloaders, get_balanced_dataloaders, preprocess_data, avg_dict, get_FFM_freqs, FourierFeatureMapping

from math import ceil
import matplotlib.pyplot as plt
import time


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

def process_invalid_patches(patches, thresh=0.5):
    '''Takes in a tensor of patches and returns a tensor of ones and zeros, with ones signaling an invalid patch so that the cost can be set appropriately.

    Args:
        - patches:
            Tensor of shape [B, C, H, W]
    
    Returns:    
        - invalid:
            Tensor of shape [B]
    '''
    invalid_channels = patches[:, -1, :, :]
    invalid_vals = torch.sum(invalid_channels, dim=(1,2))/(patches.shape[-2]*patches.shape[-1])
    invalid_flags = invalid_vals > thresh

    return invalid_flags

def produce_costmap(model, maps, map_metadata, crop_params, vel=None, fourier_freqs=None):
    '''Returns a costmap using a trained model from a maps dict.

    Args:
        - model:
            nn.Module object, Torch model used for cost inference.
        - maps: 
            A dictionary of maps (as would go into TerrainMap) defined as follows:
            {
                'rgb_map': Tensor(C,H,W) where C=3 corresponding to RGB values,
                'height_map': Tensor(C,H,W) where C=5 corresponding to min,     max, mean, std, invalid_mask where 1's correspond to invalid cells
            }
        - map_metadata: 
            Information about the map in metric space defined as follows: 
            {
                'height': map_height [m],
                'width': map_width [m],
                'resolution': resolution [m],
                'origin': origin [m]
            }
        - crop_params:
            Dictionary containing information about the output crops     
            {
                'crop_size': [Float, Float] # size in meters of the patch to obtain below the robot,
                'output_size': [Int, Int] # Size of output image in pixels
            }
        - vel:
            Float of unnormalized velocity at which we want to query the costmap. If name of the model is not CostVelModel or CostFourierVelModel, this should be None.
        - fourier_freqs:
            Tensor of fourier frequencies used in the CostFourierVelModel. If the name of the model is different, this should be None.
            
    Returns:
        - costmap:
            Tensor of dimensions as given by the map_metadata: (height/resolution, width/resolution) containing inferred costmap from learned model.
    '''
    # import pdb;pdb.set_trace()
    device = "cuda" # "cuda" if torch.cuda.is_available() else "cpu"
    tm = TerrainMap(maps=maps, map_metadata=map_metadata, device=device)


    # Get tensor of all map poses to be queried
    map_height = int(map_metadata['height']/map_metadata['resolution'])
    map_width = int(map_metadata['width']/map_metadata['resolution'])
    stride = 20
    x_pixels = torch.arange(0, map_height, stride)
    y_pixels = torch.arange(0, map_width, stride)
    x_poses = x_pixels*map_metadata['resolution']+map_metadata["origin"][0]
    y_poses = y_pixels*map_metadata['resolution']+map_metadata["origin"][1]
    all_poses = torch.stack(torch.meshgrid(x_poses, y_poses, indexing="ij"), dim=-1).view(-1, 2)
    # Append orientations
    all_poses = torch.cat([all_poses, torch.zeros(all_poses.shape[0], 1)], dim=-1).to(device).detach()

    num_cells = all_poses.shape[0]
    batch_size = 4
    num_batches = ceil(num_cells/batch_size)
    batch_starts = [(k)*batch_size for k in range(num_batches)]
    batch_ends   = [min(((k+1)*batch_size), num_cells) for k in range(num_batches)]

    all_costs = []
    # Query all map poses from TerrainMap
    # fig = plt.figure()
    # front_img_ax = fig.add_subplot(111)
    for b in range(num_batches):
        # if b % 100 == 0:
        #     print(f"Evaluating batch {b}/{num_batches}")
        # import pdb;pdb.set_trace()
        patches = tm.get_crop_batch(poses=all_poses[batch_starts[b]:batch_ends[b]], crop_params=crop_params)
        # print(f"Shape of patches: {patches.shape}")
        invalid_flags = process_invalid_patches(patches, thresh=0.5)
        # rgb_maps, height_maps = patches_to_imgs(patches)
        # front_img_ax.clear() 
        # front_img_ax.imshow(rgb_maps[0])
        # p = all_poses[batch_starts[b]:batch_ends[b]]
        # front_img_ax.set_title(f"Element {b}. Looking at pose {p}")
        # Pass all map patches to network
        # import pdb;pdb.set_trace()
        input_data = {}
        # import pdb;pdb.set_trace()
        input_data['patches'] = patches.cuda()
        if vel is not None:
            vels_vec = (torch.ones(patches.shape[0], 1) * vel/20.0).cuda()
        else:
            vels_vec = None
        if fourier_freqs is not None:
            fourier_freqs = fourier_freqs.cuda()
            fourier_vels = (FourierFeatureMapping(vels_vec, fourier_freqs)).cuda()
        else:
            fourier_vels = None
        input_data['vels'] = vels_vec
        input_data['fourier_vels'] = fourier_vels
        costs = model(input_data).detach()
        costs[invalid_flags] = 0.5 # TODO Uncomment this line if you want to set high costs to invalid areas
        # import pdb;pdb.set_trace()
        if len(costs.shape) > 1:
            costs = costs.squeeze()
        if len(costs.shape) < 1:
            costs = costs.view(-1)
        all_costs.append(costs)
        # plt.pause(0.1)
    
    all_costs = torch.cat(all_costs, 0)
    # Reshape cost predictions into costmap
    # import pdb;pdb.set_trace()
    reduced_costmap = all_costs.view(1, 1, x_pixels.shape[0], y_pixels.shape[0])

    costmap = torch.nn.functional.interpolate(reduced_costmap, size=(map_height,map_width), mode='bilinear', align_corners=True)
    # costmap = reduced_costmap 

    costmap = costmap.squeeze()
    
    # costmap = all_costs.view(map_height, map_width)
    costmap = costmap.cpu().numpy()

    return costmap



def rosmsgs_to_maps(rgbmap, heightmap):
    '''Converts input rgbmaps and heightmaps from numpy arrays incoming from ros msgs to tensors that can be passed into produce_costmap.

    Args:
        - rgbmap:
            HxWx3 Uint8 array containing rgbmap input from ros topic.
        - heightmap:
            HxWx4 Float array containing the following info about heightmap: min, max, mean, std.
    Returns:
        - maps:
            Dictionary containing two tensors:
            {
                'rgb_map': Tensor(C,H,W) where C=3 corresponding to RGB values,
                'height_map': Tensor(C,H,W) where C=5 corresponding to min,     max, mean, std, invalid_mask where 1's correspond to invalid cells
            }
    '''
    ## First, convert rgbmap to tensor
    img_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    rgb_map_tensor = img_transform(rgbmap.astype(np.uint8))
    # Change axes so that map is aligned with robot-centric coordinates
    rgb_map_tensor = rgb_map_tensor.permute(0,2,1)

    ## Now, convert heightmap to tensor
    hm = torch.from_numpy(heightmap)
    hm_nan = torch.isnan(hm).any(dim=-1, keepdim=True) | (hm > 1e5).any(dim=-1, keepdim=True) | (hm < -1e5).any(dim=-1, keepdim=True)
    hm = torch.nan_to_num(hm, nan=0.0, posinf=2, neginf=-2)
    hm = torch.clamp(hm, min=-2, max=2)
    hm = (hm - (-2))/(2 - (-2))
    hm = torch.cat([hm, hm_nan], dim=-1)
    hm = hm.permute(2,0,1)
    height_map_tensor = hm.permute(0,2,1)

    maps = {
            'rgb_map':rgb_map_tensor,
            'height_map':height_map_tensor
        }

    return maps


def local_path_to_pixels(local_path, map_metadata):
    '''Returns the pixel locations of a local_path in the costmap.
    
    Args:
        - local_path:
            Nx3 array of local path obtained from odometry
        - map_metadata:
            Dictionary containing metadata for costmap. Has the following structure:
            {
                'height': map_height [m],
                'width': map_width [m],
                'resolution': resolution [m],
                'origin': origin [m]
            }
    '''

    x_positions = local_path[:,0]
    y_positions = local_path[:,1]

    x_pixels = ((x_positions - map_metadata["origin"][0])/map_metadata["resolution"]).long()
    y_pixels = ((y_positions - map_metadata["origin"][1])/map_metadata["resolution"]).long()

    return x_pixels, y_pixels
