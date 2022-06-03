from learned_cost_map.trainer.utils import get_dataloaders, preprocess_data
from learned_cost_map.terrain_utils.terrain_map_tartandrive import get_local_path
from learned_cost_map.utils.costmap_utils import local_path_to_pixels
from learned_cost_map.viz.visualize_model_output import tensor_to_img

import torch
import matplotlib.pyplot as plt


odom_tensor_1 = torch.Tensor([
        [ 5.0000e+00,  5.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          7.0711e-01,  7.0711e-01,  1.0000e+00,  1.0000e+00,  1.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 5.0000e+00,  6.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          7.0711e-01,  7.0711e-01,  1.0000e+00,  1.0000e+00,  1.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 6.0000e+00,  7.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          3.8268e-01,  9.2388e-01,  1.0000e+00,  1.0000e+00,  1.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 7.0000e+00,  8.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e-00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 8.0000e+00,  8.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e-00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,
          0.0000e+00,  0.0000e+00,  0.0000e+00]])



local_path_1 = torch.Tensor([
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 2.5000e-01,  0.0000e+00,  0.0000e+00],
        [ 5.0000e-01,  0.0000e+00,  0.0000e+00],
        [ 7.5000e-01,  0.0000e+00,  0.0000e+00],
        [ 1.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 1.5000e+00,  0.0000e+00,  0.0000e+00],
        [ 2.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 3.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 3.5000e+00,  0.0000e+00,  0.0000e+00],
        [ 4.0000e+00,  0.0000e+00,  0.0000e+00]])

local_path_2 = torch.Tensor([
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  2.5000e-01,  0.0000e+00],
        [ 0.0000e+00,  5.0000e-01,  0.0000e+00],
        [ 0.0000e+00,  7.5000e-01,  0.0000e+00],
        [ 0.0000e+00,  1.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  1.5000e+00,  0.0000e+00],
        [ 0.0000e+00,  2.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  3.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  3.5000e+00,  0.0000e+00],
        [ 0.0000e+00,  4.0000e+00,  0.0000e+00]])

local_path_3 = torch.Tensor([
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 2.5000e-01,  0.0000e+00,  6.8772e-04],
        [ 5.0000e-01,  0.0000e+00,  8.1342e-04],
        [ 7.5000e-01,  0.0000e+00,  8.1778e-05],
        [ 1.0000e+00,  0.0000e+00, -7.6205e-04],
        [ 1.5000e+00,  0.0000e+00, -3.1771e-03],
        [ 2.0000e+00,  0.0000e+00, -4.2513e-03],
        [ 3.0000e+00,  0.0000e+00, -5.3097e-03],
        [ 3.5000e+00,  0.0000e+00, -5.9950e-03],
        [ 4.0000e+00,  0.0000e+00, -5.9518e-03]])

local_path_4 = torch.Tensor([
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 2.5000e-01,  0.0000e+00,  0.0000e+00],
        [ 5.0000e-01,  0.0000e+00,  0.0000e+00],
        [ 7.5000e-01,  0.0000e+00,  0.0000e+00],
        [ 1.0000e+00,  0.0000e+00,  0.0000e+00],
        [ 1.5000e+00,  5.0000e-01,  0.0000e+00],
        [ 2.0000e+00,  1.0000e+00,  0.0000e+00],
        [ 3.0000e+00,  1.5000e+00,  0.0000e+00],
        [ 3.5000e+00,  1.7500e+00,  0.0000e+00],
        [ 4.0000e+00,  2.0000e+00,  0.0000e+00]])


def main():
    batch_size = 10
    seq_length = 1
    data_root_dir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories'
    train_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    val_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    num_workers = 4
    shuffle_train = False
    shuffle_val = False
    train_loader, val_loader = get_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val)
    print("Fine")

    fig = plt.figure()
    path_ax = fig.add_subplot(111)
    for i, data_dict in enumerate(train_loader):
        # Get rgb map
        rgb_map_tensor = data_dict["rgbmap"][0,0].permute(0,2,1)
        rgb_map_array = tensor_to_img(rgb_map_tensor)

        # Get local odom to plot path:
        odom_tensor = odom_tensor_1 #data_dict["odom"][0]
        # import pdb;pdb.set_trace()
        vels = torch.linalg.norm(odom_tensor[:,7:10], dim=1)
        print(vels)
        local_path = get_local_path(odom_tensor)
        map_metadata = {
                'height': 12.0,
                'width': 12.0,
                'resolution': 0.02,
                'origin': [-2.0, -6.0]
            }
        # import pdb;pdb.set_trace()
        path_pix_x, path_pix_y = local_path_to_pixels(local_path_1, map_metadata)
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