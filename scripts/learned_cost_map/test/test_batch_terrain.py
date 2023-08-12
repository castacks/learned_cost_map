import numpy as np
import cv2

from torch.utils.data import Dataset
import torch
import yaml
from torchvision import transforms as T
from learned_cost_map.terrain_utils.terrain_map_tartandrive import TerrainMap, get_local_path


if __name__=="__main__":
    maps = torch.load("maps_dict.pt")
    map_config = "/home/mateo/phoenix_ws/src/learned_cost_map/configs/map_params.yaml"
    with open(map_config, "r") as file:
        map_info = yaml.safe_load(file)
    map_metadata = map_info["map_metadata"]
    crop_params = map_info["crop_params"]

    device = "cpu"
    tm = TerrainMap(maps=maps, map_metadata=map_metadata, device=device)
    local_path = torch.load("local_path.pt")

    patches = tm.get_crop_batch(local_path, crop_params)

    # import pdb;pdb.set_trace()