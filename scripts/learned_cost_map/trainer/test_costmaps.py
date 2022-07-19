from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from learned_cost_map.trainer.model import CostModel, CostVelModel, CostFourierVelModel, CostFourierVelModelEfficientNet, CostFourierVelModelRGB, CostFourierVelModelSmall, CostModelEfficientNet
from learned_cost_map.utils.costmap_utils import produce_costmap

from learned_cost_map.trainer.utils import get_dataloaders, get_balanced_dataloaders, preprocess_data, avg_dict, get_FFM_freqs, tensor_to_img
from math import ceil
import matplotlib.pyplot as plt
import time
import yaml


def main(batch_size = 256, seq_length = 10, model_name="CostModel", saved_model=None, saved_freqs=None, vel=None, map_config=None, costmap_config=None, mlp_size=512):
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
    train_loader, val_loader = get_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val, map_config, augment_data=False)

    pretrained=False
    embedding_size=512

    fourier_freqs = None
    if model_name=="CostModel":
        model = CostModel(input_channels=8, output_size=1)
    elif model_name=="CostVelModel":
        model = CostVelModel(input_channels=8, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1)
    elif model_name=="CostFourierVelModel":
        model = CostFourierVelModel(input_channels=8, ff_size=16, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1, pretrained=pretrained)
        fourier_freqs = torch.load(saved_freqs)
    elif model_name=="CostModelEfficientNet":
        model = CostModelEfficientNet(input_channels=8, output_size=1)
    elif model_name=="CostFourierVelModelEfficientNet":
        model = CostFourierVelModelEfficientNet(input_channels=8, ff_size=16, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1)
        fourier_freqs = torch.load(saved_freqs)
    elif model_name=="CostFourierVelModelSmall":
        model = CostFourierVelModelSmall(input_channels=8, ff_size=16, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1)
        fourier_freqs = torch.load(saved_freqs)
    elif model_name=="CostFourierVelModelRGB":
        model = CostFourierVelModelRGB(input_channels=3, ff_size=16, embedding_size=embedding_size, mlp_size=mlp_size, output_size=1)
        fourier_freqs = torch.load(saved_freqs)
    else:
        raise NotImplementedError()

    # Load trained model to produce costmaps
    # model = CostModel(input_channels=8, output_size=1).cuda()
    model.load_state_dict(torch.load(saved_model))
    model.cuda()
    model.eval()

    with open(map_config, "r") as file:
        map_info = yaml.safe_load(file)
    map_metadata = map_info["map_metadata"]
    crop_params = map_info["crop_params"]

    with open(costmap_config, "r") as file:
        costmap_params = yaml.safe_load(file)
    costmap_batch_size = costmap_params["batch_size"]
    costmap_stride = costmap_params["stride"]


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
        costmap = produce_costmap(model, maps, map_metadata, crop_params, costmap_batch_size, costmap_stride, vel, fourier_freqs)
        print(f"Time to produce costmap: {time.time() - before} s.")
        front_img_ax.clear()
        rgb_map_ax.clear()
        costmap_ax.clear()

        front_img_ax.imshow(color_img_array)
        front_img_ax.set_title("Front facing image")
        rgb_map_ax.imshow(rgb_map_array, origin="lower")
        rgb_map_ax.set_title("RGB map")
        costmap_im = costmap_ax.imshow(np.swapaxes(costmap, 0, 1), vmin=0.0, vmax=1.0, cmap="viridis", origin="lower")
        cb = plt.colorbar(costmap_im, shrink=0.4)
        costmap_ax.set_title("Learned Costmap")
        # if i==0:
        #     plt.pause(5)
        plt.pause(0.1)
        cb.remove()

if __name__ == '__main__':
    # Run training loop

    # saved_model = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_uni_aug_l2/epoch_50.pt"
    # saved_freqs = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_uni_aug_l2/fourier_freqs.pt"

    # saved_model = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_bal_aug_l2/epoch_50.pt"
    # saved_freqs = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_bal_aug_l2/fourier_freqs.pt"

    # saved_model = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModelEfficientNet_uni_aug_l2/epoch_50.pt"
    # saved_freqs = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModelEfficientNet_uni_aug_l2/fourier_freqs.pt"


    ## MLP 32
    # saved_model = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_MLP_32_2/epoch_50.pt"
    # saved_freqs = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_MLP_32_2/fourier_freqs.pt"
    # map_config = "/home/mateo/phoenix_ws/src/learned_cost_map/configs/map_params.yaml"
    # costmap_config = "/home/mateo/phoenix_ws/src/learned_cost_map/configs/output_costmap_params.yaml"

    ## MLP 128
    # saved_model = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_MLP_128_1/epoch_50.pt"
    # saved_freqs = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_MLP_128_1/fourier_freqs.pt"
    # map_config = "/home/mateo/phoenix_ws/src/learned_cost_map/configs/map_params.yaml"
    # costmap_config = "/home/mateo/phoenix_ws/src/learned_cost_map/configs/output_costmap_params.yaml"

    ## MLP 512
    # saved_model = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_lr_3e-4_g_99e-1_bal_aug_l2_scale_10.0/epoch_50.pt"
    # saved_freqs = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModel_lr_3e-4_g_99e-1_bal_aug_l2_scale_10.0/fourier_freqs.pt"
    # map_config = "/home/mateo/phoenix_ws/src/learned_cost_map/configs/map_params.yaml"
    # costmap_config = "/home/mateo/phoenix_ws/src/learned_cost_map/configs/output_costmap_params.yaml"

    ## Smaller ResNet
    saved_model = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModelSmall_lr_3e-4_g_99e-1_bal_aug_l2_scale_10.0_0/epoch_50.pt"
    saved_freqs = "/home/mateo/phoenix_ws/src/learned_cost_map/models/train_CostFourierVelModelSmall_lr_3e-4_g_99e-1_bal_aug_l2_scale_10.0_0/fourier_freqs.pt"
    map_config = "/home/mateo/phoenix_ws/src/learned_cost_map/configs/map_params.yaml"
    costmap_config = "/home/mateo/phoenix_ws/src/learned_cost_map/configs/output_costmap_params.yaml"

    model_name = "CostFourierVelModelSmall"
    vel = 5.0
    mlp_size=512
    main(batch_size = 1, seq_length = 1, model_name=model_name, saved_model=saved_model, saved_freqs=saved_freqs, vel=vel, map_config=map_config, costmap_config=costmap_config, mlp_size=mlp_size)