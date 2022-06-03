import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from learned_cost_map.trainer.utils import get_dataloaders, preprocess_data
from learned_cost_map.trainer.model import CostModel, CostVelModel, CostFourierVelModel
from learned_cost_map.terrain_utils.terrain_map_tartandrive import get_local_path
from learned_cost_map.utils.costmap_utils import local_path_to_pixels
from learned_cost_map.utils.util import quat_to_yaw


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
    if patches_tensor.shape[0] == 1 or len(patches_tensor.shape) < 4:
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

def main(model_name, saved_model, saved_freqs):
    batch_size = 1
    seq_length = 1
    # data_root_dir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories'
    # train_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    # val_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    data_root_dir = '/home/mateo/Data/SARA/TartanDriveCostTrain'
    train_split = '/home/mateo/Data/SARA/TartanDriveCostTrain/tartandrive_train.txt'
    val_split = '/home/mateo/Data/SARA/TartanDriveCostTrain/tartandrive_train.txt'
    num_workers = 4
    shuffle_train = False
    shuffle_val = False
    train_loader, val_loader = get_dataloaders(batch_size, seq_length, data_root_dir, train_split, val_split, num_workers, shuffle_train, shuffle_val)

    fig = plt.figure()
    front_facing_ax = fig.add_subplot(221)
    top_down_ax = fig.add_subplot(222)
    paths_ax = fig.add_subplot(223)
    pred_costmap_ax = fig.add_subplot(224)


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

    model.load_state_dict(torch.load(saved_model))
    model.cuda()
    model.eval()

    for i, data_dict in enumerate(val_loader):
        # Get cost from neural net
        input, labels = preprocess_data(data_dict, fourier_freqs)
        pred_costs = model(input).detach().cpu().squeeze()

        vel = 20.0*torch.mean(input["vels"]).cpu().item()
        print(vel)
        fig.suptitle(f"Learned Cost on Training Data. Model: {model_name}. Vel: {vel:.2f}")

        # Get front facing
        rgb_front = data_dict["imgc"][0,0]
        rgb_front_array = tensor_to_img(rgb_front)

        # Get rgb map
        rgb_map_tensor = data_dict["rgbmap"][0,0].permute(0,2,1)
        rgb_map_array = tensor_to_img(rgb_map_tensor)

        # Plot patches obtained from dataloader
        patches = data_dict["patches"][0]
        masks = data_dict["masks"][0]

        rgb_maps, height_maps = patches_to_imgs(patches)
        costs = data_dict["cost"][0]
        empty_map = torch.zeros(rgb_map_array.shape[:-1])

        for i, mask in enumerate(masks):
            cost = int(costs[i]*255)
            pixel_list = mask.view(-1, 2)
            empty_map[pixel_list[:,0], pixel_list[:,1]] = cost

        empty_map = empty_map.cpu().numpy()

        # Get local odom to plot path:
        odom_tensor = data_dict["odom"][0]
        odom_tartanvo = data_dict["odom_tartanvo"][0]
        
        # import pdb;pdb.set_trace()
        local_path = get_local_path(odom_tensor)
        local_path_tartanvo = get_local_path(odom_tartanvo)

        ## GPS odom is 90 degrees rotated NED To FLU
        local_path = torch.index_select(local_path, 1, torch.LongTensor([1, 0, 2]))
        local_path[:,1] = -local_path[:,1]

        map_metadata = {
                'height': 12.0,
                'width': 12.0,
                'resolution': 0.02,
                'origin': [-2.0, -6.0]
            }
        path_pix_x, path_pix_y = local_path_to_pixels(local_path, map_metadata)
        gt_costmap = torch.zeros(rgb_map_array.shape[:-1])
        pred_costmap = torch.zeros(rgb_map_array.shape[:-1])


        for j, mask in enumerate(masks):
            gt_cost = int(costs[j]*255)
            pred_cost = int(pred_costs[j]*255)
            pixel_list = mask.view(-1, 2)
            gt_costmap[pixel_list[:,0], pixel_list[:,1]] = gt_cost
            pred_costmap[pixel_list[:,0], pixel_list[:,1]] = pred_cost

        print(f"Path coords:")
        print(f"X coords: {local_path[:,0]}")
        print(f"Y coords: {local_path[:,1]}")
        print(f"Yaws (rad): {local_path[:,2]}")

        print(f"Path pixels:")
        print(f"X coords: {path_pix_x}")
        print(f"Y coords: {path_pix_y}")


        for ix, iy in zip(path_pix_x, path_pix_y):
            gt_costmap[ix-3:ix+4, iy-3:iy+4] = 255 # TODO: Change to known cost value
            pred_costmap[ix-3:ix+4:, iy-3:iy+4] = 255 # TODO: Change to known cost value

        gt_costmap = gt_costmap.cpu().numpy().astype(np.uint8)
        pred_costmap = pred_costmap.cpu().numpy().astype(np.uint8)

        front_facing_ax.clear()
        top_down_ax.clear()
        paths_ax.clear()
        pred_costmap_ax.clear()

        
        # pred_costmap_im = pred_costmap_ax.imshow(pred_costmap, origin="lower", vmin=0.0, vmax=255.0)
        pred_costmap_im = pred_costmap_ax.imshow(np.swapaxes(pred_costmap, 0, 1), origin="lower", vmin=0.0, vmax=255.0)
        pred_costmap_ax.set_title("Predicted costmap")
        # cb_pred = plt.colorbar(pred_costmap_im, shrink=0.4)



        front_facing_ax.imshow(rgb_front_array)
        front_facing_ax.set_title("Front facing")

        top_down_ax.imshow(rgb_map_array, origin="lower")
        top_down_ax.set_title("RGB map")

        paths_ax.plot(odom_tensor[:,0]-odom_tensor[0,0], odom_tensor[:,1]-odom_tensor[0,1], label="Raw Odom")
        paths_ax.plot(local_path[:,0], local_path[:,1], label="GPS Path")
        paths_ax.plot(local_path_tartanvo[:,0], local_path_tartanvo[:,1], label="TartanVO Path")
        paths_ax.set_xlabel("X axis")
        paths_ax.set_ylabel("Y axis")
        paths_ax.set_xlim([-6.0, 6.0])
        paths_ax.set_ylim([-6.0, 6.0])
        paths_ax.set_aspect(1.0)
        paths_ax.legend()


        plt.subplots_adjust(wspace=1.5)
        if i == 0:
            plt.pause(5)
        plt.pause(1)
        # plt.show()
        # cb_gt.remove()
        # cb_pred.remove()

if __name__=="__main__":
    model_name = "CostFourierVelModel"
    # saved_model = "/home/mateo/models/train_CostModel2/epoch_50.pt"
    # saved_model = "/home/mateo/models/train_CostVelModel/epoch_50.pt"
    # saved_freqs = None

    saved_model = "/home/mateo/models/train_CostFourierVelModel/epoch_50.pt"
    saved_freqs = "/home/mateo/models/train_CostFourierVelModel/fourier_freqs.pt"
    main(model_name, saved_model, saved_freqs)