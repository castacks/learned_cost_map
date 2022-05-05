import numpy as np
import cv2

from torch.utils.data import Dataset
import torch
from torchvision import transforms as T
from learned_cost_map.terrain_utils.terrain_map_tartandrive import TerrainMap, get_local_path


if __name__=="__main__":
    maps = torch.load("maps_dict.pt")

    map_height = 12.0 # [m]
    map_width  = 12.0 # [m]
    resolution = 0.02
    origin     = [-2.0, -6.0]

    crop_width = 2.0  # in meters
    crop_size = [crop_width, crop_width]
    output_size = [224, 224]

    # TODO. Make sure the two dicts below are populated using from input parameters
    map_metadata = {
        'height': map_height,
        'width': map_width,
        'resolution': resolution,
        'origin': origin
    }

    crop_params ={
        'crop_size': crop_size,
        'output_size': output_size
    }

    device = "cpu"
    tm = TerrainMap(maps=maps, map_metadata=map_metadata, device=device)
    local_path = torch.load("local_path.pt")

    patches = tm.get_crop_batch(local_path, crop_params)

    # import pdb;pdb.set_trace()