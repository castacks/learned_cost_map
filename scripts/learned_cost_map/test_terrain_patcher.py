import os 
import numpy as np
from learned_cost_map.terrain_utils.terrain_map_tartandrive import TerrainMap, get_local_path
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt

bgr_to_rgb = lambda img: img[[2,1,0],:,:] 
    
transform_to_img = T.Compose([
    T.Normalize(mean = [0., 0., 0.], std  = [1/0.229, 1/0.224, 1/0.225]),
    T.Normalize(mean = [-0.485, -0.456, -0.406], std  = [1., 1., 1.]),
    # T.Lambda(bgr_to_rgb),
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


def load_data(traj_dir, seq_len):
    '''Fake data loader for now to get data for Cost Learning.

    For now, it just returns heightmaps, rgbmaps, odom, and costs
    
    '''
    hm_dir = os.path.join(traj_dir, "height_map")
    rgb_dir = os.path.join(traj_dir, "rgb_map")
    odom_dir = os.path.join(traj_dir, "tartanvo_odom")
    cost_dir = os.path.join(traj_dir, "cost")
    
    # Get filepaths to load in all data
    hm_files = [os.path.join(hm_dir,x) for x in sorted(os.listdir(hm_dir)) if x.endswith(".npy")]
    rgb_files = [os.path.join(rgb_dir,x) for x in sorted(os.listdir(rgb_dir)) if x.endswith(".npy")]
    odom_fp = os.path.join(odom_dir, "poses.npy")
    cost_fp = os.path.join(cost_dir, "cost.npy")

    # Load height maps
    hm_data = []
    for i, file in enumerate(hm_files):
        hm_data.append(np.load(file))
    hm_data = np.stack(hm_data, axis=0)

    # Load rgb maps
    rgb_data = []
    for i, file in enumerate(rgb_files):
        rgb_data.append(np.load(file))
    rgb_data = np.stack(rgb_data, axis=0)

    # Load odom data
    odom_data = np.load(odom_fp)

    # Load cost data
    cost_data = np.load(cost_fp)

    heightmaps = hm_data[50:50+seq_len]
    rgbmaps    = rgb_data[50:50+seq_len]
    odom       = odom_data[50:50+seq_len]
    costs      = cost_data[50:50+seq_len]


    return heightmaps, rgbmaps, odom, costs

def get_crops(heightmaps, rgbmaps, odom):
        '''Returns (patches, costs)'''
        # Set up TerrainMap object
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

        min_height = 0
        max_height = 2

        # Extract maps at k=0, switch to be channel first, and change height and width order to convert to robot-centric coordinates (+x forward, +y left)
        rgb_map_tensor = torch.from_numpy(np.copy(rgbmaps[0])).permute(2,1,0) # (C,W,H)
        height_map_tensor = torch.from_numpy(np.copy(heightmaps[0])).permute(2,1,0) # (C,W,H)

        maps = {
            'rgb_map':rgb_map_tensor,
            'height_map':height_map_tensor
        }

        device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
        tm = TerrainMap(maps=maps, map_metadata=map_metadata, device=device)

        local_path = get_local_path(torch.from_numpy(odom)).to(device)
        patches = tm.get_crop_path(local_path, crop_params)
        patches = [patch.permute(1,2,0).cpu().numpy() for patch in patches]

        for i,patch in enumerate(patches):
            # import pdb;pdb.set_trace()
            patch_np = patch[:,:,:3]
            # import pdb;pdb.set_trace()
            plt.imshow(patch_np.astype(np.uint8))
            plt.show()
        
        return patches
        

def data_transform(sample):
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
        # patches_hm = torch.clamp(patches_hm, min=-2, max=2)
        # patches_hm = (patches_hm - (-2))/(2 - (-2))
        patches_hm = torch.cat([patches_hm, patches_hm_nan], dim=-1)

        # Process rgb maps # TODO uncomment normalization
        img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        imgs = []
        for img in patches_rgb:
            img_torch = img_transform(img.astype(np.uint8))
            imgs.append(img_torch)
        patches_rgb = torch.stack(imgs,0)

        patches_hm = patches_hm.permute(0,3,1,2)
        patches = torch.cat([patches_rgb, patches_hm], dim=-3)
        sample["patches"] = patches

    return sample

if __name__ == "__main__":
    # Get data as I would from Wenshan's DataLoader
    traj_dir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories/000009'
    seq_len = 15
    heightmaps, rgbmaps, odom, costs = load_data(traj_dir, seq_len)

    patches = get_crops(heightmaps, rgbmaps, odom)

    sample = {'patches': patches}

    t_sample = data_transform(sample)

    rgb_maps, height_maps = patches_to_imgs(t_sample["patches"])

    for i,patch in enumerate(rgb_maps):
        plt.imshow(patch)
        plt.show()

    # import pdb;pdb.set_trace()