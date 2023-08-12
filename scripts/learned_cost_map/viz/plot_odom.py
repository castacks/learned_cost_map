from learned_cost_map.trainer.utils import get_dataloaders, preprocess_data
from learned_cost_map.terrain_utils.terrain_map_tartandrive import get_local_path
from learned_cost_map.utils.costmap_utils import local_path_to_pixels
from learned_cost_map.viz.visualize_model_output import tensor_to_img

import torch
import yaml
import matplotlib.pyplot as plt

def main():
    batch_size = 10
    seq_length = 1
    data_root_dir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories'
    train_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    val_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    map_config = '/home/mateo/phoenix_ws/src/learned_cost_map/configs/map_params.yaml'
    with open(map_config, "r") as file:
        map_info = yaml.safe_load(file)
    map_metadata = map_info["map_metadata"]
    crop_params = map_info["crop_params"]

    num_workers = 4
    shuffle_train = False
    shuffle_val = False
    train_loader, val_loader = get_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val, map_config)
    print("Fine")

    fig = plt.figure()
    path_ax = fig.add_subplot(111)
    for i, data_dict in enumerate(train_loader):
        # Get rgb map
        rgb_map_tensor = data_dict["rgbmap"][0,0].permute(0,2,1)
        rgb_map_array = tensor_to_img(rgb_map_tensor)

        # Get local odom to plot path:
        odom_tensor = data_dict["odom"][0]
        vels = torch.linalg.norm(odom_tensor[:,7:10], dim=1)
        print(vels)
        local_path = get_local_path(odom_tensor)

        # import pdb;pdb.set_trace()
        path_pix_x, path_pix_y = local_path_to_pixels(local_path, map_metadata)
        print(rgb_map_array.shape[:-1])
        gt_costmap = torch.zeros(rgb_map_array.shape[:-1])

        gt_costmap[path_pix_x, path_pix_y] = 255

        path_ax.clear()
        path_ax.set_xlabel("X axis")
        path_ax.set_ylabel("Y axis")
        path_ax.set_title(f"Avg speed: {torch.mean(vels).item()}")
        path_ax.imshow(gt_costmap, origin="lower")
        plt.pause(1)

if __name__=="__main__":
    main()