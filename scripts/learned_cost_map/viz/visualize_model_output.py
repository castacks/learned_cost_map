import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from learned_cost_map.trainer.utils import get_dataloaders, preprocess_data
from learned_cost_map.trainer.model import CostModel


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

def main():
    batch_size = 1
    seq_length = 10
    train_loader, val_loader = get_dataloaders(batch_size, seq_length)

    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=10, nrows=4, figure=fig)
    front_rgb_ax = fig.add_subplot(spec[0:2, 0:5])
    rgb_map_ax   = fig.add_subplot(spec[0:2,5:])
    patches_axs  = [fig.add_subplot(spec[2,i]) for i in range(5)] + [fig.add_subplot(spec[3,i]) for i in range(5)]
    height_map_ax = fig.add_subplot(spec[2:4, 5:], projection="3d")

    saved_model = "/home/mateo/learned_cost_map/scripts/learned_cost_map/trainer/models/epoch_20.pt"
    model = CostModel(input_channels=8, output_size=1).cuda()
    model.load_state_dict(torch.load(saved_model))
    model.eval()


    # img_viewer = fig.add_subplot(111, projection="3d")
    for i, data_dict in enumerate(train_loader):
        # Get cost from neural net
        x, y = preprocess_data(data_dict)
        pred_cost = model(x).squeeze()

        # import pdb;pdb.set_trace()

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
            empty_map[pixel_list[:,1], pixel_list[:,0]] = cost

        empty_map = empty_map.cpu().numpy()


        front_rgb_ax.clear()
        rgb_map_ax.clear()
        height_map_ax.clear()
        front_rgb_ax.imshow(color_img_array)
        front_rgb_ax.set_title("Front Facing Camera")
        rgb_map_ax.imshow(rgb_map_array, origin="lower")
        rgb_map_ax.imshow(empty_map, origin="lower", alpha=0.3)
        rgb_map_ax.set_xlabel("X axis")
        rgb_map_ax.set_ylabel("Y axis")
        rgb_map_ax.set_title("RGB map")
        for i,patch in enumerate(rgb_maps):
            patches_axs[i].imshow(patch, origin="lower")
            cost = costs[i]
            patches_axs[i].set_title(f"GT Cost: {cost:.2f}\nPred Cost: {pred_cost:.2f}")
        height_map_ax.plot_surface(x,y,height_map_array_low, cmap=cm.coolwarm, alpha=0.5)
        height_map_ax.plot_surface(x,y,height_map_array_high, cmap=cm.PRGn, alpha=0.5)
        height_map_ax.set_xlabel("X axis")
        height_map_ax.set_ylabel("Y axis")
        height_map_ax.set_zlabel("Z axis")
        height_map_ax.set_title("Height map")
        plt.pause(0.1)

if __name__=="__main__":
    main()