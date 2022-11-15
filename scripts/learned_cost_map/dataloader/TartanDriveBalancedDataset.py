from __future__ import print_function

import numpy as np
import cv2
import os
import random

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms as T
from torchvision.datasets import DatasetFolder
from learned_cost_map.terrain_utils.terrain_map_tartandrive import TerrainMap, get_local_path

def balanced_data_transform(sample, augment_data=False):
    # import pdb;pdb.set_trace()
    # Transform left_img=img0, right_img=img1, color_img=imgc, disparity image=disp0
    # Convert to Tensor
    # Transform to pytorch tensors, make sure they are all in CxHxW configuration
    if "imgc" in sample:
        img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        img_torch = img_transform(sample["imgc"].astype(np.uint8))
        sample["imgc"] = img_torch

        # imgs = []
        # stacked_np = np.stack([img for img in sample["imgc"]],0)
        # for img in stacked_np:
        #     img_torch = img_transform(img.astype(np.uint8))
        #     imgs.append(img_torch)
        # sample["imgc"] = torch.stack(imgs, 0)


    # Transform heightmap:
    # Convert to Tensor
    # Clamp at [-2,2]
    # Normalize so that it is between 0 and 1
    # Make sure channels go first
    if "heightmap" in sample:
        hm = torch.from_numpy(sample["heightmap"])
        # hm = torch.stack([torch.from_numpy(img) for img in hm],0)
        hm_nan = torch.isnan(hm).any(dim=-1, keepdim=True) | (hm > 1e5).any(dim=-1, keepdim=True) | (hm < -1e5).any(dim=-1, keepdim=True)
        hm = torch.nan_to_num(hm, nan=0.0, posinf=2, neginf=-2)
        hm = torch.clamp(hm, min=-2, max=2)
        hm = (hm - (-2))/(2 - (-2))
        hm = torch.cat([hm, hm_nan], dim=-1)
        # hm = hm.permute(0,3,1,2)
        hm = hm.permute(2,0,1)
        sample["heightmap"] = hm

    # Transform rgbmap:
    # Convert to Tensor
    # Normalize using ImageNet normalization
    # Make sure channels go first
    if "rgbmap" in sample:
        img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        img_torch = img_transform(sample["rgbmap"].astype(np.uint8))
        sample["rgbmap"] = img_torch

        # imgs = []
        # stacked_np = np.stack([img for img in sample["rgbmap"]],0)
        # for img in stacked_np:
        #     img_torch = img_transform(img.astype(np.uint8))
        #     imgs.append(img_torch)
        # sample["rgbmap"] = torch.stack(imgs, 0)

    # Transform cmd, odom, cost, imu to be tensors 
    if "cmd" in sample:
        sample["cmd"] = torch.from_numpy(sample["cmd"])

    if "odom" in sample:
        sample["odom"] = torch.from_numpy(sample["odom"])

    if "cost" in sample:
        sample["cost"] = torch.from_numpy(sample["cost"].reshape(1))

    if "imu" in sample:
        sample["imu"] = torch.from_numpy(sample["imu"])


    # Transform patches:
    # Convert to Tensor
    # Clamp last 4 dimensions at [-2,2]
    # import pdb;pdb.set_trace() 
    if "patches" in sample:
        patches = sample["patches"]
        stacked_np = np.stack([img for img in patches],0)
        # Process heightmaps
        patches = torch.stack([torch.from_numpy(img) for img in patches],0)
        patches_hm = patches[...,3:]
        patches_rgb = stacked_np[...,:3] #patches[...,:3]

        patches_hm_nan = torch.isnan(patches_hm).any(dim=-1, keepdim=True) | (patches_hm > 1e5).any(dim=-1, keepdim=True) | (patches_hm < -1e5).any(dim=-1, keepdim=True)
        patches_hm = torch.nan_to_num(patches_hm, nan=0.0, posinf=2, neginf=-2)
        patches_hm = torch.clamp(patches_hm, min=-2, max=2)
        patches_hm = (patches_hm - (-2))/(2 - (-2))
        patches_hm = torch.cat([patches_hm, patches_hm_nan], dim=-1)

        # Process rgb maps
        if augment_data:
            img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                T.RandomApply(torch.nn.ModuleList([
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                ]), p=0.5)
            ])
        else:
            img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        imgs = []
        for img in patches_rgb:
            img_torch = img_transform(img.astype(np.uint8))
            imgs.append(img_torch)
        patches_rgb = torch.stack(imgs,0)

        patches_hm = patches_hm.permute(0,3,1,2)
        patches = torch.cat([patches_rgb, patches_hm], dim=-3)
        

        # # Add data augmentation 
        if augment_data:
            augment_transform = T.Compose([
                T.RandomVerticalFlip(p=0.5),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])

            patches = augment_transform(patches)

        sample["patches"] = patches.squeeze()
    return sample


class BalancedTartanDrive(Dataset):
    def __init__(self, data_dir_lc, data_dir_hc, map_metadata, crop_params, transform=None, augment_data = False, high_cost_prob=None):
        self.data_dir_lc = data_dir_lc
        self.data_dir_hc = data_dir_hc
        self.transform = transform
        self.augment_data = augment_data

        self.map_metadata = map_metadata
        self.crop_params = crop_params

        cmds_dir_lc = os.path.join(self.data_dir_lc, "cmds.npy")
        costs_dir_lc = os.path.join(self.data_dir_lc, "costs.npy")
        odoms_dir_lc = os.path.join(self.data_dir_lc, "odoms.npy")

        self.cmds_lc = np.load(cmds_dir_lc)
        self.costs_lc = np.load(costs_dir_lc)
        self.odoms_lc = np.load(odoms_dir_lc)

        cmds_dir_hc = os.path.join(self.data_dir_hc, "cmds.npy")
        costs_dir_hc = os.path.join(self.data_dir_hc, "costs.npy")
        odoms_dir_hc = os.path.join(self.data_dir_hc, "odoms.npy")

        self.cmds_hc = np.load(cmds_dir_hc)
        self.costs_hc = np.load(costs_dir_hc)
        self.odoms_hc = np.load(odoms_dir_hc)

        self.N_lc = self.odoms_lc.shape[0]
        self.N_hc = self.odoms_hc.shape[0]
        self.N = self.N_lc + self.N_hc
        print(f"Total frames: {self.N}. {self.N_lc} low cost + {self.N_hc} high cost.")

        if high_cost_prob is None:
            self.high_cost_prob = 1 - self.N_hc/self.N
            print(f"Setting high cost sampling probability to 1 - {self.N_hc}/{self.N} = {self.high_cost_prob}")
        else:
            self.high_cost_prob = high_cost_prob
            print(f"Setting high cost sampling probability to {self.high_cost_prob}")
        self.all_data = self.build_indices(self.high_cost_prob)

    def __len__(self):
        return self.N

    def __getitem__(self, all_data_idx):
        sample = {}

        all_data_elem = self.all_data[all_data_idx]

        # print(f"Get item {all_data_elem} with index {all_data_idx}")

        if all_data_elem[1] == 0:
            # Low cost
            self.data_dir = self.data_dir_lc
            self.cmds = self.cmds_lc
            self.costs = self.costs_lc
            self.odoms = self.odoms_lc
        else:
            # High cost
            self.data_dir = self.data_dir_hc
            self.cmds = self.cmds_hc
            self.costs = self.costs_hc
            self.odoms = self.odoms_hc

        idx = all_data_elem[0]

        # # Format the index into the right string
        # sample["cmd"] = self.cmds[idx]
        # sample["cost"] = self.costs[idx]
        # sample["odom"] = self.odoms[idx]

        # Change to take into account new directory structure for combination of 2021 and 2022 data
        cmd_dir = ""
        sample["cost"] = np.array([0, 0]) # TODO need to actually extract this data.
        cost_dir = os.path.join(self.data_dir, "cost", f"{idx:06}.npy")
        sample["cost"] = np.load(cost_dir)[0]
        odom_dir = os.path.join(self.data_dir, "odom", f"{idx:06}.npy")
        sample["odom"] = np.load(odom_dir)[0]

        # Load images
        imgc_dir = os.path.join(self.data_dir, "image_left_color", f"{idx:06}.png")
        heightmap_dir = os.path.join(self.data_dir, "height_map", f"{idx:06}.npy")
        rgbmap_dir = os.path.join(self.data_dir, "rgb_map", f"{idx:06}.npy")

        imgc = cv2.imread(imgc_dir, cv2.IMREAD_UNCHANGED)
        heightmap = np.load(heightmap_dir)
        rgbmap = np.load(rgbmap_dir)

        sample["imgc"] = imgc
        sample["heightmap"] = heightmap
        sample["rgbmap"] = rgbmap

        patches, masks = self.get_crops(heightmaps=heightmap, rgbmaps=rgbmap, map_metadata=self.map_metadata, crop_params=self.crop_params)

        sample["patches"] = patches
        sample["masks"] = masks

        if ( self.transform is not None):
            sample = self.transform(sample, self.augment_data)

        return sample

    def get_crops(self, heightmaps, rgbmaps, map_metadata, crop_params):
        '''Returns (patches, costs)
        '''
        # Set up TerrainMap object

        min_height = 0
        max_height = 2

        # Extract maps at k=0, switch to be channel first, and change height and width order to convert to robot-centric coordinates (+x forward, +y left)
        rgb_map_tensor = torch.from_numpy(np.copy(rgbmaps)).permute(2,1,0) # (C,W,H)
        height_map_tensor = torch.from_numpy(np.copy(heightmaps)).permute(2,1,0) # (C,W,H)

        maps = {
            'rgb_map':rgb_map_tensor,
            'height_map':height_map_tensor
        }

        device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
        tm = TerrainMap(maps=maps, map_metadata=map_metadata, device=device)

        odom = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # <- Query only the patch at the origin with orientation of 0
        local_path = get_local_path(torch.from_numpy(odom)).to(device)
        
        ## GPS odom is 90 degrees rotated NED To FLU
        local_path = torch.index_select(local_path, 1, torch.LongTensor([1, 0, 2]))
        local_path[:,1] = -local_path[:,1]

        # patches = tm.get_crop_batch(local_path, crop_params)
        patches, masks = tm.get_crop_batch_and_masks(local_path, crop_params)
        patches = [patch.permute(1,2,0).cpu().numpy() for patch in patches]
        masks = masks.squeeze()
        return patches, masks

    def build_indices(self, p_hc=0.5):
        all_data = []

        # 1. Create list of tuples for each class: low cost and high cost
        low_cost_tuples = [(i, 0) for i in range(self.N_lc)]
        high_cost_tuples = [(i, 1) for i in range(self.N_hc)]

        # 2. Shuffle order of low cost and high cost tuples
        random.shuffle(low_cost_tuples)
        random.shuffle(high_cost_tuples)


        # 3. Populate all_data with probability p_hc for high_cost indices
        while (len(low_cost_tuples) > 0) or (len(high_cost_tuples) > 0):
            if len(low_cost_tuples) == 0:
                all_data.extend(high_cost_tuples)
                high_cost_tuples = []
            elif len(high_cost_tuples) == 0:
                all_data.extend(low_cost_tuples)
                low_cost_tuples = []
            else:
                rand_sample = np.random.rand()
                if rand_sample < p_hc:
                    sample = high_cost_tuples.pop()
                    all_data.append(sample)
                else:
                    sample = low_cost_tuples.pop()
                    all_data.append(sample) 
                 
        assert ((len(low_cost_tuples) == 0) and (len(high_cost_tuples) == 0)), "Did not include all samples"
        
        # print(all_data)

        return all_data

if __name__ == '__main__':

    data_dir_lc = "/home/mateo/Data/SARA/tartancost_data/tartandrive_lowcost_frames"
    data_dir_hc = "/home/mateo/Data/SARA/tartancost_data/tartandrive_highcost_frames"
    dataset = BalancedTartanDrive(data_dir_lc, \
                                  data_dir_hc, \
                                  transform=balanced_data_transform, \
                                  augment_data = False)

    print('Dataset length: ',len(dataset))
    # import pdb;pdb.set_trace()
    max_sample = 1755
    # for k in range(0, max_sample, 1):
    #     print(f"Printing sample {k}")
    #     sample = dataset[k]
    #     # import pdb;pdb.set_trace()

    #     patch = sample['patches'].unsqueeze(0)
    #     rgb_maps, height_maps = patches_to_imgs(patch)

    #     plt.imshow(rgb_maps[0])
    #     cost = sample['cost'].item()
    #     plt.title(f"Cost is {cost}")
    #     plt.show()
    # print('---')



    # train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

    # for i, sample in enumerate(train_loader):
    #     # import pdb;pdb.set_trace()
    #     print(f"Sample {i}")
    #     print(sample["cost"].shape)

    #     patches = sample['patches']
    #     rgb_maps, height_maps = patches_to_imgs(patches)
    #     for j in range(rgb_maps.shape[0]):
    #         plt.imshow(rgb_maps[j])
    #         cost = sample['cost'][j].item()
    #         plt.title(f"Cost is {cost}")
    #         plt.show()
