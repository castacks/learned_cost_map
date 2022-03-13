import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
# from learned_cost_map.path_transform import TrajLoader, PathTransform
# from learned_cost_map.extract_patch import PatchTransform
# from learned_cost_map.util import dict_to, quat_to_yaw, dict_map

from path_transform import TrajLoader, PathTransform
from extract_patch import PatchTransform
from util import dict_to, quat_to_yaw, dict_map


def get_global_local_path(state, next_state):
    current_p = state[:,:3]
    current_q = state[:,3:]
    current_yaw = quat_to_yaw(current_q)
    
    next_p = next_state[:,:3]
    next_q = next_state[:,3:]
    next_yaw = quat_to_yaw(next_q)

    dx_global = next_p[:, 0] - current_p[:, 0]
    dy_global = next_p[:, 1] - current_p[:, 1]
    dyaw = next_yaw - current_yaw

    # Original
    dx = torch.cos(-current_yaw[[0]])*dx_global - torch.sin(-current_yaw[[0]])*dy_global
    dy = torch.sin(-current_yaw[[0]])*dx_global + torch.cos(-current_yaw[[0]])*dy_global

    # dy = -torch.cos(-current_yaw[[0]])*dx_global + torch.sin(-current_yaw[[0]])*dy_global
    # dx = torch.sin(-current_yaw[[0]])*dx_global + torch.cos(-current_yaw[[0]])*dy_global

    x_local = torch.cumsum(dx, dim=0).view(-1, 1)
    y_local = torch.cumsum(dy, dim=0).view(-1, 1)
    yaw_local = torch.cumsum(dyaw, dim=0).view(-1,1)
    local_path = torch.cat((x_local, y_local, yaw_local), 1)

    x_global = torch.cumsum(dx_global, dim=0).view(-1, 1)
    y_global = torch.cumsum(dy_global, dim=0).view(-1, 1)
    yaw_global = torch.cumsum(dyaw, dim=0).view(-1,1)  # Check
    global_path = torch.cat((x_global, y_global, yaw_global), 1)

    return global_path, local_path


def get_local_path(state, next_state):
    """Builds local path which starts at first state (position, yaw). 
    Does this by querying difference between next-state and state, then accumulate.
    This is helpful when we split into subtrajectories?
    """
    current_p = state[:,:3] # position
    current_q = state[:,3:] # quaterion 
    current_yaw = quat_to_yaw(current_q)
    
    next_p = next_state[:,:3]
    next_q = next_state[:,3:]
    next_yaw = quat_to_yaw(next_q)

    dx_global = next_p[:, 0] - current_p[:, 0]
    dy_global = next_p[:, 1] - current_p[:, 1]
    dyaw = next_yaw - current_yaw

    dx = torch.cos(-current_yaw[[0]])*dx_global - torch.sin(-current_yaw[[0]])*dy_global
    dy = torch.sin(-current_yaw[[0]])*dx_global + torch.cos(-current_yaw[[0]])*dy_global


    x_local = torch.cumsum(dx, dim=0).view(-1, 1)
    y_local = torch.cumsum(dy, dim=0).view(-1, 1)
    yaw_local = torch.cumsum(dyaw, dim=0).view(-1,1)
    local_path = torch.cat((x_local, y_local, yaw_local), 1)

    

    return local_path



class TerrainMap:
    def __init__(self, maps = {}, map_metadata = {}, device='cpu'):
        self.maps = maps
        self.device = device 

        self.map_metadata = {}
        self.maps = {}

        for k, v in self.default_map_metadata.items():
            self.map_metadata[k] = torch.tensor(map_metadata[k] if k in map_metadata.keys() else self.default_map_metadata[k])

        for k,v in self.default_maps.items():
            self.maps[k] = torch.tensor(maps[k] if k in maps.keys() else self.default_maps[k])

        self.map_metadata = dict_to(self.map_metadata, self.device)
        self.maps = dict_to(self.maps, self.device)

        self.maps_tensor = torch.cat([self.maps['rgb_map'], self.maps['height_map_low'], self.maps['height_map_high']], dim=0)

        self.num_channels = self.maps_tensor.shape[0]

        self.resolution = self.map_metadata['resolution']
        self.origin = self.map_metadata['origin']

    def get_crop(self, pose, crop_params):
        '''Obtain crop of map at a given pose.

        Args:
            - pose:
                Tensor of [x, y, yaw] pose for crop to obtain
            - crop_params:
                Dictionary of params for crop:

                {'crop_size': [crop_size_x, crop_size_y],
                 'output_size': [output_size_x, output_size_y]}

                crop_size is in meters, output_size is in pixels

        Returns:
            - crop:
                Nxcrop_size tensor of crop at input pose, where N is the number of channels in self.maps
        '''

        crop_size = [int(c/self.resolution) for c in crop_params['crop_size']]

        p_x = int((pose[0] - self.origin[0])/self.resolution)
        p_y = int((pose[1] - self.origin[1])/self.resolution)
        position = [p_x, p_y]
        yaw = pose[2].item()

        getPatch = PatchTransform(position, yaw, crop_size, crop_params['output_size'], on_top=True)

        crop = getPatch(self.maps_tensor)

        return crop


    def get_crop_path(self, path, crop_params):
        '''Obtain an NxCxHxW tensor of crops for a given path.
        
        Args:
            - path:
                Nx3 tensor of poses, where a pose is represented as [x, y, yaw].
            - crop_params:
                Dictionary of params for crop:

                {'crop_size': [crop_size_x, crop_size_y],
                 'output_size': [output_size_x, output_size_y]}

                crop_size is in meters, output_size is in pixels

        Returns:
            - crops:
                Tensor of NxCxHxW of crops at poses on the path, where C is the number of channels in self.maps and N is the number of points in the path
        '''

        crops = []

        for i in range(path.shape[0]):
            pose = path[i,:]
            crops.append(self.get_crop(pose, crop_params).view(1, self.num_channels, *crop_params['output_size']))

        crops = torch.cat(crops, dim=0)
        return crops

    def get_crop_batch(self, poses, crop_params):
        '''Obtain an NxCxHxW tensor of crops for a given path.

        Procedure:
        1. Get initial meshgrid for crop in metric space centered around the origin
        2. Apply affine transform to all the crop positions (to the entire meshgrid)
        3. Center around the right origin and rescale to obtain pixel coordinates
        4. 
        
        Args:
            - path:
                Nx3 tensor of poses, where N is the number of poses to evaluate and each pose is represented as [x, y, yaw].
            - crop_params:
                Dictionary of params for crop:

                {'crop_size': [crop_size_x, crop_size_y],
                 'output_size': [output_size_x, output_size_y]}

                crop_size is in meters, output_size is in pixels

        Returns:
            - crops:
                Tensor of NxCxHxW of crops at poses on the path, where C is the number of channels in self.maps and N is the number of points in the path
        '''
        #For reference, all tensors will obey the following axis convention:
        # [batch x crop_x x crop_y x transform_x x transform_y]
        # Also note that this is working in metric space. As such, the terrain map axes are flipped relative

#        import pdb;pdb.set_trace()

        ## Create initial crop template in metric space centered around (0,0) to generate all pixel values

        crop_xs = torch.linspace(-crop_params['crop_size'][0]/2., crop_params['crop_size'][0]/2., crop_params['output_size'][1]).to(self.device)
        crop_ys = torch.linspace(-crop_params['crop_size'][1]/2., crop_params['crop_size'][1]/2., crop_params['output_size'][1]).to(self.device)
        crop_positions = torch.stack(torch.meshgrid(crop_xs, crop_ys, indexing="ij"), dim=-1) # HxWx2 tensor

        translations = poses[:, :2]  # Nx2 tensor, where each row corresponds to [x, y] position in metric space
        rotations = torch.stack([poses[:, 2].cos(), -poses[:, 2].sin(), poses[:, 2].sin(), poses[:, 2].cos()], dim=-1)  # Nx4 tensor where each row corresponds to [cos(theta), -sin(theta), sin(theta), cos(theta)]
        
        ## Reshape tensors to perform batch tensor multiplication. 

        # The goal is to obtain a tensor of size [B, H, W, 2], where B is the batch size, H and W are the dimensions fo the image, and 2 corresponds to the actual x,y positions. To do this, we need to rotate and then translate every pair of points in the meshgrid. In batch multiplication, only the last two dimensions matter. That is, usually we need the following dimensions to do matrix multiplication: (m,n) x (n,p) -> (m,p). In batch multiplication, the last two dimensions of each array need to line up as mentioned above, and the earlier dimensions get broadcasted (more details in the torch matmul page). Here, we will reshape rotations to have shape [B,1,1,2,2] where B corresponds to batch size, the two dimensions with size 1 are there so that we can broadcast with the [H,W] dimensions in crop_positions, and the last two dimensions with size 2 reshape the each row in rotations into a rotation matrix that can left multiply a position to transform it. The output of torch.matmul(rotations, crop_positions) will be a [B,H,W,2,1] tensor. We will reshape translations to be a [B,1,1,2,1] vector so that we can add it to this output and obtain a tensor of size [B,H,W,2,1], which we will then squeeze to obtain our final tensor of size [B,H,W,2]
        
        rotations = rotations.view(-1, 1, 1, 2, 2) #[B x 1 x 1 x 2 x 2]
        crop_positions = crop_positions.view(1, *crop_params['output_size'], 2, 1) #[1 x H x W x 2 x 1]
        translations = translations.view(-1, 1, 1, 2, 1) #[B x 1 x 1 x 2 x 1]
        
        
        # Apply each transform to all crop positions (res = [B x H x W x 2])
        crop_positions_transformed = (torch.matmul(rotations, crop_positions) + translations).squeeze()

        # Obtain actual pixel coordinates
        map_origin = self.map_metadata['origin'].view(1, 1, 1, 2)
        pixel_coordinates = ((crop_positions_transformed - map_origin) / self.map_metadata['resolution']).long()

        pixel_coordinates_flipped = pixel_coordinates.swapaxes(-2,-3)

        # Obtain maximum and minimum values of map to later filter out of bounds pixels
        map_p_low = torch.tensor([0, 0]).to(self.device).view(1, 1, 1, 2)
        map_p_high = torch.tensor(self.maps_tensor.shape[1:]).to(self.device).view(1, 1, 1, 2)
        invalid_mask = (pixel_coordinates < map_p_low).any(dim=-1) | (pixel_coordinates >= map_p_high).any(dim=-1)

#        import matplotlib.pyplot as plt
#        for i in torch.arange(0, 100, 20):
#            plt.scatter(pixel_coordinates[i,...,0].flatten().cpu(), pixel_coordinates[i,...,1].flatten().cpu(), s=1.)
#            print(poses[i].cpu())
#        plt.show()

        #Indexing method: set all invalid idxs to a valid one (i.e. 0), index, then mask out the results

        #TODO: Per-channel fill value
        fill_value = 0
        pixel_coordinates[invalid_mask]=0
        pixel_coordinates_flipped[invalid_mask] = 0


        pxlist = pixel_coordinates.view(-1, 2)
        pxlist_flipped = pixel_coordinates_flipped.reshape(-1,2)

        #[B x C x W x H]
        # import pdb;pdb.set_trace()
        values_temp = self.maps_tensor[:,pxlist_flipped[:,0], pxlist_flipped[:,1]]  # Notice axes are flipped to account for terrain body centric coordinates.
        values_temp = values_temp.view(self.maps_tensor.shape[0], poses.shape[0], *crop_params['output_size']).swapaxes(0, 1)

        values = self.maps_tensor.swapaxes(-1, -2)[:, pxlist[:, 0], pxlist[:, 1]].view(self.maps_tensor.shape[0], poses.shape[0], *crop_params['output_size']).swapaxes(0, 1)

#        k1 = invalid_mask.unsqueeze(1).repeat(1, self.maps_tensor.shape[0], 1, 1)
#        values[k1] = fill_value
        values_1 = values.swapaxes(-1,-2)
        import pdb;pdb.set_trace()

        k1 = invalid_mask.unsqueeze(1).float()
        values = (1.-k1)*values + k1*fill_value

        return values.swapaxes(-1, -2)
        

    def get_labeled_crops(self, trajectory, crop_params):
        
        crop_width = crop_params['crop_size'][0] # Assume crop width and height are equal
        # Extract path from trajectory -> use odom to get path
        # NOTE: Path coordinates are in robot centric FLU metric coordinates
        state = trajectory['observation']['state']
        next_state = trajectory['next_observation']['state']

        cost = trajectory['observation']['traversability_cost']
        local_path = get_local_path(state, next_state)
   
        print("local_path shape", local_path.shape)
        # Convert metric coordinates to pixel coordinates to be sampled
        crops = self.get_crop_path(local_path, crop_params)
        fpv_images = trajectory['observation']['image_rgb']

        return crops, cost, fpv_images, local_path


    def get_rgb_map(self):
        return self.maps['rgb_map']
        

    def get_occupancy_grid(self):
        pass

    default_map_metadata = {
        'height': 10.0,
        'width': 10.0,
        'resolution': 0.05,
        'origin': [0.0, -5.0],
        # 'fields': ['rgb_map_r', 'rgb_map_g', 'rgb_map_r', 'height_map_low', 'height_map_high']
    }

    default_maps = {
        'rgb_map': torch.zeros(3, 200, 200),
        'height_map_low': torch.zeros(3, 200, 200),
        'height_map_high': torch.zeros(3, 200, 200)
    }

    def convert_local_to_pixel(self, local_path):
        """Convert local_path to pixel space.
        Local_path: N x 3 numpy array (x, y, theta)
        Output: Nx3 Numpy array
        """
        x = local_path[:,0]
        y = local_path[:,1]
        theta = local_path[:,2]
        pixel_x = (x/self.resolution.numpy()).astype(int)
        pixel_y = ((y - self.origin[1].numpy())/self.resolution.numpy()).astype(int)
        pixel_xyt = np.vstack((pixel_x, pixel_y, theta)).T
        return pixel_xyt

    def visualize_rgb_map(self, ax=None):
        """Plot RGB Map. Assumes its first 3 dimensions of self.maps_tensor."""
        rgb_map = self.maps_tensor.permute(1,2,0).numpy()[:,:,:3] # get rgb map
        if ax is None: # make a new figure
            plt.figure()
            plt.imshow(rgb_map, origin="lower")
        else:
            # print("AX IS NOT NONE, not implemented")
            ax.imshow(rgb_map, origin="lower")
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")

        
        if ax is None:
            plt.show()

    def visualize_map_with_path(self, local_path, ax=None):
        """Visualize map overlaid with local path."""

        # TODO: add animate ON
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            # Convert from local space to pixel space
            pixel_xyt = self.convert_local_to_pixel(local_path.numpy())
            # Plot
            self.visualize_rgb_map(ax)
            ax.scatter(pixel_xyt[:,0], pixel_xyt[:,1], color='r', s=1)

            plt.show()

        else:
            print("AX IS NOT NONE, not implemented")        

    def animate_map_crops_fpv(self, crops, costs, fpv_images, local_path):
        fig = plt.figure()
        fig.suptitle('Data visualizer')
        map_ax = fig.add_subplot(131)
        patch_ax = fig.add_subplot(132)
        fpv_ax = fig.add_subplot(133)

        pixel_xyt = self.convert_local_to_pixel(local_path.numpy())

        for i in np.arange(crops.shape[0]):
            map_ax.clear()
            patch_ax.clear()
            fpv_ax.clear()

            self.visualize_rgb_map(map_ax)
            map_ax.scatter(pixel_xyt[:,0], pixel_xyt[:,1], color='r', s=1) # plot whole traj
            map_ax.scatter(pixel_xyt[i,0], pixel_xyt[i,1], color='b') # plot current odom
            
            img = crops[i].permute(1,2,0).numpy()[:,:,:3]
            patch_ax.imshow(img)
            patch_ax.set_title(f"Looking at patch {i}/{crops.shape[0]}. \n Cost: {costs[i]:.4f}")

            fpv_ax.imshow(fpv_images[i,:3].permute(1,2,0).numpy())

            plt.pause(0.01)
            if i==0:
                plt.pause(1)        


if __name__ == "__main__":
    
    # remake_traj_loaders = True # if false, loads trajloader object 

    device = "gpu"
    #  Directory with .pt files from rosbag_to_dataset
    base_dir = '/home/cherie/Desktop/mateo-files/CostBags'
    

    # Given directory of .pt files, make list of datasets (self.trajectories/self.trajectories_stamped)
    hi1 = time.time()
    tl = TrajLoader(base_dir, device) # Most time is spent here
    print("hi1", time.time() - hi1)
    hi2 = time.time()
    traj = tl.getTrajectory()[0]
    print("hi2", time.time() - hi2)
    slicer = lambda x: x
    traj_all = dict_map(traj, slicer) #! What is dict_map for?

    # For now, hardcode time index where map is fully populated.
    slicer = lambda x: x[0:1]
    traj_sliced = dict_map(traj, slicer) 

    # Constants on map size
    map_height = 10.0
    map_width = 10.0
    resolution = 0.05

    crop_width = 2  # in meters
    crop_size = [crop_width, crop_width]
    output_size = [224, 224]

    # Instantiate Terrain Map, seems like it can handle multiple maps

    # Get maps to make TerrainMap object (all seems to be completed map), which data collection is operated over
    # NOTE: Permutation step is really important to have x and y coordinates in the right coordinate frame
    rgb_map = traj_sliced['observation']['rgb_map_inflate'][0].permute(0,2,1)
    height_map_low = traj_sliced['observation']['heightmap_low'][0].permute(0,2,1)
    height_map_high = traj_sliced['observation']['heightmap_high'][0].permute(0,2,1)

    map_metadata = {
        'height': map_height,
        'width': map_width,
        'resolution': resolution
    }

    maps = {
        'rgb_map': rgb_map,
        'height_map_low': height_map_low,
        'height_map_high':height_map_high
    } 

    crop_params ={
        'crop_size': crop_size,
        'output_size': output_size
    }

    tm = TerrainMap(map_metadata=map_metadata, maps=maps)

    # Get dataset {crops, cost}
    crops, costs, fpv_images, local_path = tm.get_labeled_crops(traj_all, crop_params)
    pixel_xyt = tm.convert_local_to_pixel(local_path.numpy())

    # Visualizations
    # tm.visualize_rgb_map()
    # tm.visualize_map_with_path(local_path, ax=None)
    # tm.animate_map_crops_fpv(crops, costs, fpv_images, local_path)

    # Visualize Trajectory
    fig = plt.figure()
    fig.suptitle('Observation visualizer')
    map_ax = fig.add_subplot(131)
    rgb_map_ax = fig.add_subplot(132)
    fpv_ax = fig.add_subplot(133)
    rgb_map_lists = []
    local_pose_lists = []
    ts_list = []
    for i in range(traj_all['observation']['rgb_map'].shape[0]):
        map_ax.clear()
        rgb_map_ax.clear()
        fpv_ax.clear()

        tm.visualize_rgb_map(map_ax)
        map_ax.scatter(pixel_xyt[:,0], pixel_xyt[:,1], color='r', s=1) # plot whole traj
        map_ax.scatter(pixel_xyt[i,0], pixel_xyt[i,1], color='b') # plot current odom
        pixel_dx = np.cos(pixel_xyt[i,2])*20
        pixel_dy = np.sin(pixel_xyt[i,2])*20
        map_ax.arrow(pixel_xyt[i,0]-0.5*pixel_dx, pixel_xyt[i,1]-0.5*pixel_dy, pixel_dx, pixel_dy, color="blue", head_width=10)

        rgb_map_inflate_permute = traj_all['observation']['rgb_map_inflate'][i].permute(0,2,1) # flipping x, y coord
        rgb_map_inflate = rgb_map_inflate_permute.permute(1,2,0).numpy()[:,:,:3]
        rgb_map_ax.imshow(rgb_map_inflate)
        fpv_ax.imshow(traj_all['observation']['image_rgb'][i,:3].permute(1,2,0).numpy())
        rgb_map_ax.set_title(i)
        plt.pause(0.01)

        cur_pixel = pixel_xyt[i,:]
        if i==0:
            plt.pause(1)
            local_pose_lists.append(cur_pixel)
            rgb_map_lists.append(rgb_map_inflate)
            ts_list.append(i)
        if i==50:
            local_pose_lists.append(cur_pixel)
            rgb_map_lists.append(rgb_map_inflate)
            ts_list.append(i)
            print(cur_pixel)
        if i == 100:
            local_pose_lists.append(cur_pixel)
            rgb_map_lists.append(rgb_map_inflate)
            ts_list.append(i)
            print(cur_pixel)

            np.save("rgb_map_list", rgb_map_lists)
            np.save("local_pose_list", local_pose_lists)
            np.save("ts_list", ts_list)
    # torch.save(crops, '/home/mateo/SARA/src/sara_ws/src/traversability_cost/scripts/crops.pt')