from collections import OrderedDict
import numpy as np
import os
import torch
from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim
from learned_cost_map.trainer.model import CostModel

from learned_cost_map.trainer.utils import *
from math import ceil
import matplotlib.pyplot as plt
import time


def produce_costmap(model, maps, map_metadata, crop_params):
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
    stride = 10
    x_pixels = torch.arange(0, map_height, stride)
    y_pixels = torch.arange(0, map_width, stride)
    all_poses = torch.stack(torch.meshgrid(x_pixels, y_pixels, indexing="ij"), dim=-1).view(-1, 2)
    # Append orientations
    all_poses = torch.cat([all_poses, torch.zeros(all_poses.shape[0], 1)], dim=-1).to(device).detach()

    num_cells = all_poses.shape[0]
    batch_size = 256
    num_batches = ceil(num_cells/batch_size)
    batch_starts = [(k)*batch_size for k in range(num_batches)]
    batch_ends   = [min(((k+1)*batch_size), num_cells) for k in range(num_batches)]

    all_costs = []
    # Query all map poses from TerrainMap
    for b in range(num_batches):
        # if b % 100 == 0:
        #     print(f"Evaluating batch {b}/{num_batches}")
        patches = tm.get_crop_batch(poses=all_poses[batch_starts[b]:batch_ends[b]], crop_params=crop_params)

        # Pass all map patches to network
        costs = model(patches).detach()
        all_costs.append(costs.squeeze())
    all_costs = torch.cat(all_costs, 0)
    # Reshape cost predictions into costmap
    # import pdb;pdb.set_trace()
    reduced_costmap = all_costs.view(1, 1, x_pixels.shape[0], y_pixels.shape[0])

    costmap = torch.nn.functional.interpolate(reduced_costmap, size=(map_height,map_width), mode='bilinear', align_corners=True)

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