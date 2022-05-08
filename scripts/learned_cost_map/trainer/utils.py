import numpy as np


from torch.utils.data import DataLoader
from torchvision import transforms as T
from learned_cost_map.terrain_utils.terrain_map_tartandrive import TerrainMap, get_local_path
from learned_cost_map.dataloader.TartanDriveDataset import DatasetBase, data_transform

def get_dataloaders(batch_size, seq_length):
    data_root_dir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories'
    train_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    val_split = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'

    # datatypes = "img0,img1,imgc,disp0,heightmap,rgbmap,cmd,odom,cost,patches,imu"
    # base_mod_lengths = [1,1,1,1,1,1,1,1,1,1,10]

    datatypes = "imgc,heightmap,rgbmap,odom,cost,patches"
    base_mod_lengths = [1,1,1,1,1,1]
    modality_lengths = [seq_length*l for l in base_mod_lengths]

    train_set = DatasetBase(train_split,
                            dataroot= data_root_dir,
                            datatypes = datatypes,
                            modalitylens = modality_lengths,
                            transform=data_transform,
                            imu_freq = 10,
                            frame_skip = 0, 
                            frame_stride=5)

    val_set = DatasetBase(val_split,
                          dataroot= data_root_dir,
                          datatypes = datatypes,
                          modalitylens = modality_lengths,
                          transform=data_transform,
                          imu_freq = 10,
                          frame_skip = 0, 
                          frame_stride=5)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader

def preprocess_data(input_dict):
    x = input_dict["patches"].view(-1, *input_dict["patches"].shape[-3:])
    y = input_dict["cost"].view(-1)
    return x.to('cuda'), y.to('cuda')

def avg_dict(all_metrics):
    keys = all_metrics[0].keys()
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = np.mean([all_metrics[i][key].cpu().detach().numpy() for i in range(len(all_metrics))])
    return avg_metrics


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