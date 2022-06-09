import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from learned_cost_map.trainer.utils import get_dataloaders, preprocess_data
from learned_cost_map.utils.costmap_utils import bgr_to_rgb, transform_to_img, tensor_to_img, tensor_to_heightmap, patches_to_imgs


def main():
    batch_size = 10
    seq_length = 1
    data_root_dir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories'
    train_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    val_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    num_workers = 4
    shuffle_train = False
    shuffle_val = False
    train_loader, val_loader = get_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val, augment_data=False)

    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=10, nrows=4, figure=fig)
    front_rgb_ax = fig.add_subplot(spec[0:2, 0:5])
    rgb_map_ax   = fig.add_subplot(spec[0:2,5:])
    patches_axs  = [fig.add_subplot(spec[2,i]) for i in range(5)] + [fig.add_subplot(spec[3,i]) for i in range(5)]
    height_map_ax = fig.add_subplot(spec[2:4, 5:], projection="3d")


    # img_viewer = fig.add_subplot(111, projection="3d")
    # import pdb;pdb.set_trace()
    for i, data_dict in enumerate(train_loader):
        # Plot RGB front facing image
        color_img_tensor = data_dict["imgc"][0,0]
        color_img_array  = tensor_to_img(color_img_tensor)

        rgb_map_tensor = data_dict["rgbmap"][0,0].permute(0,2,1)
        rgb_map_array = tensor_to_img(rgb_map_tensor)

        height_map_tensor = data_dict["heightmap"][0,0].permute(0,2,1)
        height_map_array_low = tensor_to_heightmap(height_map_tensor)[:,:,0]
        height_map_array_high = tensor_to_heightmap(height_map_tensor)[:,:,1]
        x,y = np.meshgrid(range(height_map_array_high.shape[0]), range(height_map_array_high.shape[1]))


        # Plot patches obtained from dataloader
        # import pdb;pdb.set_trace()
        patches = data_dict["patches"][0]
        masks = data_dict["masks"][0]

        # import pdb;pdb.set_trace()
        rgb_maps, height_maps = patches_to_imgs(patches)
        # for i,patch in enumerate(rgb_maps):
        #     print(f"Looking at patch {i}/{len(rgb_maps)}")
        #     plt.imshow(patch)
        #     plt.colorbar()
        #     plt.show()
        costs = data_dict["cost"][0]
        # import pdb;pdb.set_trace()
        empty_map = torch.zeros(rgb_map_array.shape[:-1])

        for i, mask in enumerate(masks):
            cost = int(costs[i]*255)
            pixel_list = mask.view(-1, 2)
            empty_map[pixel_list[:,0], pixel_list[:,1]] = cost

        empty_map = empty_map.cpu().numpy()


        front_rgb_ax.clear()
        rgb_map_ax.clear()
        height_map_ax.clear()
        front_rgb_ax.imshow(color_img_array)
        front_rgb_ax.set_title("Front Facing Camera")
        rgb_map_ax.imshow(np.swapaxes(rgb_map_array, 0, 1), origin="lower")  
        rgb_map_ax.imshow(np.swapaxes(empty_map, 0, 1), origin="lower", alpha=0.3)
        rgb_map_ax.set_xlabel("X axis")
        rgb_map_ax.set_ylabel("Y axis")
        rgb_map_ax.set_title("RGB map")
        for i,patch in enumerate(rgb_maps):
            patches_axs[i].imshow(patch, origin="lower")
            cost = costs[i]
            patches_axs[i].set_title(f"Cost: {cost:.2f}")
        height_map_ax.plot_surface(x,y,height_map_array_low, cmap=cm.coolwarm, alpha=0.5)
        height_map_ax.plot_surface(x,y,height_map_array_high, cmap=cm.PRGn, alpha=0.5)
        height_map_ax.set_xlabel("X axis")
        height_map_ax.set_ylabel("Y axis")
        height_map_ax.set_zlabel("Z axis")
        height_map_ax.set_title("Height map")
        plt.pause(0.1)

if __name__=="__main__":
    main()