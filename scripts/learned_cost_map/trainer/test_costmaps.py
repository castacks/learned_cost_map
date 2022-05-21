from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from learned_cost_map.trainer.model import CostModel, CostVelModel, CostFourierVelModel

from learned_cost_map.trainer.utils import *
from math import ceil
import matplotlib.pyplot as plt
import time


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
    stride = 10
    x_pixels = torch.arange(0, map_height, stride)
    y_pixels = torch.arange(0, map_width, stride)
    x_poses = x_pixels*map_metadata['resolution']+map_metadata["origin"][0]
    y_poses = y_pixels*map_metadata['resolution']+map_metadata["origin"][0]
    all_poses = torch.stack(torch.meshgrid(x_poses, y_poses, indexing="ij"), dim=-1).view(-1, 2)
    # Append orientations
    all_poses = torch.cat([all_poses, torch.zeros(all_poses.shape[0], 1)], dim=-1).to(device).detach()

    num_cells = all_poses.shape[0]
    batch_size = 256
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
        # costs = torch.rand_like(costs)
        all_costs.append(costs.squeeze())
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


def main(batch_size = 256, seq_length = 10, model_name="CostModel", saved_model=None, saved_freqs=None, vel=None):
    # Set up dataloaders to visualize costmaps
    # data_root_dir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories'
    # train_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    # val_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    data_root_dir = '/home/mateo/Data/SARA/TartanDriveCostTrain'
    train_split = '/home/mateo/Data/SARA/TartanDriveCostTrain/tartandrive_train.txt'
    val_split = '/home/mateo/Data/SARA/TartanDriveCostTrain/tartandrive_train.txt'
    num_workers = 1
    shuffle_train = False
    shuffle_val = False
    train_loader, val_loader = get_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val)

    # import pdb;pdb.set_trace()
    fourier_freqs = None
    if model_name=="CostModel":
        model = CostModel(input_channels=8, output_size=1)
    elif model_name=="CostVelModel":
        model = CostVelModel(input_channels=8, embedding_size=512, output_size=1)
    elif model_name=="CostFourierVelModel":
        model = CostFourierVelModel(input_channels=8, ff_size=16, embedding_size=512, output_size=1)
        fourier_freqs = torch.load(saved_freqs)
    else:
        raise NotImplementedError()

    # Load trained model to produce costmaps
    # model = CostModel(input_channels=8, output_size=1).cuda()
    model.load_state_dict(torch.load(saved_model))
    model.cuda()
    model.eval()

    # Define map metadata so that we know how many cells we need to query to produce costmap
    map_height = 12.0 # [m]
    map_width  = 12.0 # [m]
    resolution = 0.02
    # origin     = [-2.0, -6.0]
    origin     = [-6.0, -2.0]

    map_metadata = {
        'height': map_height,
        'width': map_width,
        'resolution': resolution,
        'origin': origin
    }

    crop_width = 2.0  # in meters
    crop_size = [crop_width, crop_width]
    output_size = [64, 64]

    crop_params ={
        'crop_size': crop_size,
        'output_size': output_size
    }

    fig = plt.figure()
    fig.suptitle(f"Learned Costmap. Model: {model_name}. Vel: {vel:.2f}")
    front_img_ax = fig.add_subplot(131)
    rgb_map_ax = fig.add_subplot(132)
    costmap_ax = fig.add_subplot(133)
    for i, data_dict in enumerate(val_loader):
        color_img_tensor = data_dict["imgc"][0, 0]
        color_img_array  = tensor_to_img(color_img_tensor)
        rgb_map_tensor = data_dict["rgbmap"][0,0].permute(0,2,1)
        rgb_map_array = tensor_to_img(rgb_map_tensor)
        height_map_tensor = data_dict["heightmap"][0,0].permute(0,2,1)

        maps = {
            'rgb_map':rgb_map_tensor,
            'height_map':height_map_tensor
        }

        print("Producing costmap")
        before = time.time()
        costmap = produce_costmap(model, maps, map_metadata, crop_params, vel, fourier_freqs)
        print(f"Time to produce costmap: {time.time() - before} s.")
        front_img_ax.clear()
        rgb_map_ax.clear()
        costmap_ax.clear()

        front_img_ax.imshow(color_img_array)
        front_img_ax.set_title("Front facing image")
        rgb_map_ax.imshow(rgb_map_array, origin="lower")
        rgb_map_ax.set_title("RGB map")
        costmap_im = costmap_ax.imshow(costmap, vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
        cb = plt.colorbar(costmap_im, shrink=0.4)
        costmap_ax.set_title("Learned Costmap")
        if i==0:
            plt.pause(5)
        plt.pause(0.1)
        cb.remove()

if __name__ == '__main__':
    # Run training loop
    # saved_model = "models/epoch_20.pt"
    # saved_model = "/home/mateo/models/train500/epoch_35.pt"

    # saved_model = "/home/mateo/models/train_CostModel2/epoch_50.pt"
    # vel = 1.0
    # main(batch_size = 1, seq_length = 1, model_name="CostModel", saved_model=saved_model)

    # saved_model = "/home/mateo/models/train_CostVelModel/epoch_50.pt"
    # vel = 1.0
    # main(batch_size = 1, seq_length = 1, model_name="CostVelModel", saved_model=saved_model, vel=vel)

    saved_model = "/home/mateo/models/train_CostFourierVelModel/epoch_50.pt"
    saved_freqs = "/home/mateo/models/train_CostFourierVelModel/fourier_freqs.pt"
    vel = 10.0
    main(batch_size = 1, seq_length = 1, model_name="CostFourierVelModel", saved_model=saved_model, saved_freqs=saved_freqs, vel=vel)