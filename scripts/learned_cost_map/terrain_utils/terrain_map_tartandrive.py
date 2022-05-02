# terrain_map_tartandrive
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from learned_cost_map.terrain_utils.extract_patch import PatchTransform
from learned_cost_map.utils.util import dict_to, quat_to_yaw, dict_map

# from path_transform import TrajLoader, PathTransform
# from extract_patch import PatchTransform
# from util import dict_to, quat_to_yaw, dict_map


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


def get_local_path(odom):
    current_p = odom[:,:3]
    current_q = odom[:,3:]
    current_yaw = quat_to_yaw(current_q)
    
    next_odom = torch.cat([odom, odom[[-1]]], dim=0)[1:]
    next_p = next_odom[:,:3]
    next_q = next_odom[:,3:]
    next_yaw = quat_to_yaw(next_q)

    dx_global = next_p[:, 0] - current_p[:, 0]
    dy_global = next_p[:, 1] - current_p[:, 1]
    dyaw = next_yaw - current_yaw

    dx = torch.cos(-current_yaw[[0]])*dx_global - torch.sin(-current_yaw[[0]])*dy_global
    dy = torch.sin(-current_yaw[[0]])*dx_global + torch.cos(-current_yaw[[0]])*dy_global


    x_local = torch.cumsum(dx, dim=0).view(-1, 1)
    y_local = torch.cumsum(dy, dim=0).view(-1, 1)
    yaw_local = torch.cumsum(dyaw, dim=0).view(-1,1)
    local_path = torch.cat((x_local, y_local, yaw_local), 1)[:-1]

    local_path = torch.cat([torch.Tensor([[0.0, 0.0, 0.0]]), local_path], 0)

    return local_path



class TerrainMap:
    def __init__(self, maps = {}, map_metadata = {}, device='cpu'):
        self.maps = maps
        self.map_metadata = map_metadata
        self.device = device 

        for k,v in self.maps.items():
            if torch.is_tensor(v):
                self.maps[k] = v.to(self.device)
        for k,v in self.map_metadata.items():
            if torch.is_tensor(v):
                self.maps[k] = v.to(self.device)

        self.maps_tensor = torch.cat([self.maps['rgb_map'], self.maps['height_map']], dim=0)

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
            crop = self.get_crop(pose, crop_params).view(1, self.num_channels, *crop_params['output_size'])
            crops.append(crop)

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
        
        import pdb;pdb.set_trace()
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
        # values_1 = values.swapaxes(-1,-2)
        # import pdb;pdb.set_trace()

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
   

        # Convert metric coordinates to pixel coordinates to be sampled
        crops = self.get_crop_path(local_path, crop_params)


        return crops, cost


        # DEBUG
        print(f"Crops shape: {crops.shape}")

        original = tm.maps_tensor.permute(1,2,0).numpy()[:,:,:3]
        fig = plt.figure()
        fig.suptitle('Cost visualizer')
        img_viewer = fig.add_subplot(121)
        patch_viewer = fig.add_subplot(122)

        for i in range(crops.shape[0]):
            img = crops[i].permute(1,2,0).numpy()[:,:,:3]

            img_viewer.clear()
            patch_viewer.clear()

            x = local_path[i,0]
            y = local_path[i,1]
            theta = local_path[i,2]
            theta_deg = theta*180/np.pi
            pixel_x = int((x/self.resolution))
            pixel_y = int((y - self.origin[1])/self.resolution)

            pixel_dx = int(crop_width/self.resolution)*np.cos(theta)
            pixel_dy = int(crop_width/self.resolution)*np.sin(theta)

            img_viewer.plot(pixel_x, pixel_y, marker='o', color='red')
            
            img_viewer.arrow(pixel_x-0.5*pixel_dx, pixel_y-0.5*pixel_dy, pixel_dx, pixel_dy, color="red")

            pixel_crop_width = int(crop_width/self.resolution)
            corners = 0.5 * np.array([[pixel_crop_width, pixel_crop_width],
                                      [pixel_crop_width, -pixel_crop_width], [-pixel_crop_width, -pixel_crop_width], [-pixel_crop_width, pixel_crop_width], [pixel_crop_width, pixel_crop_width]])
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            print(f"Original corners: {corners}")
            print(f"Corners expanded dims: {np.expand_dims(corners, axis=2)}")
            print(f"square matmul: {np.matmul(rot, np.expand_dims(corners, axis=2)).shape}")
            square = np.matmul(rot, np.expand_dims(corners, axis=2))[:, :, 0] + np.array([pixel_x, pixel_y])
            img_viewer.plot(square[:, 0], square[:, 1], c='r')

            img_viewer.set_title(f"Current pose: x={pixel_x} y={pixel_y}, yaw={(theta_deg%360.):.2f}")
            img_viewer.set_xlabel("X axis")
            img_viewer.set_ylabel("Y axis")
            img_viewer.imshow(original, origin="lower")
            

            patch_viewer.imshow(img)
            patch_viewer.set_title(f"Looking at patch {i}/{crops.shape[0]}. Cost: {cost[i]:.4f}")
            plt.pause(0.01)
            if i == 0:
                plt.pause(15)


        # END OF DEBUG

        

        
        # Downsample path if downsample=True -> get indices
        # Extract IMU data for these indices to evaluate cost
        # Evaluate cost using IMU data
        # Extract patches at these points on the path
        # Return patches and labels

        return crops, cost

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


if __name__ == "__main__":
    base_dir = '/home/mateo/Data/SARA/CostBags'
    device = "gpu"

