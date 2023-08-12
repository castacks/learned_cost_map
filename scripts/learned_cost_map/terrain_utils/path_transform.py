import torch 
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from functools import reduce


from learned_cost_map.util import dict_map
# from util import dict_map

def quat_to_yaw(quat):
    """
    Convert a quaternion (as [x, y, z, w]) to yaw
    """
    if len(quat.shape) < 2:
        return quat_to_yaw(quat.unsqueeze(0)).squeeze()

    return torch.atan2(2 * (quat[:, 3]*quat[:, 2] + quat[:, 0]*quat[:, 1]), 1 - 2 * (quat[:, 1]**2 + quat[:, 2]**2))

class TrajLoader:
    '''
    Class that loads torch datasets given a directory into a list of datasets, where each item in the list is a dataset for a given pt file in the directory.  
    '''
    def __init__(self, base_dir=None, device='cpu'):
        self.base_dir = base_dir
        self.device = device
        self.filepath_list = None

        if self.base_dir is not None:
            # print(f"Reading trajectories from path: {self.base_dir}")
            self.filepath_list = [f for f in os.listdir(self.base_dir) if (path.isfile(path.join(self.base_dir, f)) and f[-3:] == '.pt')]
            # print(f"filepaths: {self.filepath_list}")
            self.complete_filepaths = [path.join(self.base_dir, self.filepath_list[i]) for i in range(len(self.filepath_list))]
        else:
            raise AssertionError("Need to provide a valid base directory.")

        self.trajectories = [torch.load(fp) for fp in self.complete_filepaths]
        self.trajectories_stamped = [self.timestampTraj(traj) for traj in self.trajectories]
        
        # print(self.trajectories_stamped[0])

        # print("self.trajectories: ")
        # print(self.trajectories_stamped[0]['dt'])
    def timestampTraj(self, traj):
        '''
        Takes a trajectory (a dictionary of torch tensors) and adds a field that keeps track of the cumulative time that has passed since the start of the trajectory
        '''
        traj_dt = traj["dt"]
        cum_dt = torch.cumsum(traj_dt, dim=0)
        traj["local_t"] = cum_dt
        return traj

    def getTrajectory(self):
        return self.trajectories

class PathTransform:
    def __init__(self, trajectory, map_height, map_width, resolution):
        self.trajectory = trajectory
        # State is an Nx7 array with the following fields:
        # [pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w]
        self.state = self.trajectory['observation']['state']
        self.next_state = self.trajectory['next_observation']['state']

        # self.global_path = self.extract_path_from_odom(self.state)

        self.map_width = map_width
        self.map_height = map_height
        self.resolution = resolution

        # self.local_path = self.get_local_path(self.global_path)
        self.global_path, self.local_path = self.get_global_local_path(self.state, self.next_state)
        # self.plot_path(self.global_path)
        # self.plot_path(self.local_path)

        self.FLU_path = self.draw_path(self.local_path, self.map_height, self.map_width, self.resolution)

    # def extract_path_from_odom(self, state):
        
    #     # Extract yaws from quaternions
    #     yaws = quat_to_yaw(state[:,-4:]).view(-1, 1)

    #     # Create 2D state representation (x, y, yaw)
    #     state_2D = torch.cat((state[:,:2], yaws), 1)

    #     return state_2D

    # def get_local_path(self, path):
    #     local_path = path - path[0,:]

    #     return local_path

    def get_global_local_path(self, state, next_state):
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
        dx = torch.cos(-np.pi/2-current_yaw[[0]])*dx_global - torch.sin(-np.pi/2-current_yaw[[0]])*dy_global
        dy = torch.sin(-np.pi/2-current_yaw[[0]])*dx_global + torch.cos(-np.pi/2-current_yaw[[0]])*dy_global

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

        
        plt.plot(x_local, y_local, label='local')
        plt.plot(x_global, y_global, label='global')
        plt.xlim([0, 10])
        plt.ylim([-5, 5])
        # plt.plot(next_p[:, 0]-current_p[0, 0], next_p[:, 1]-current_p[0, 1], label='global')
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.legend()
        plt.gca().set_aspect('equal')
        plt.show()
        return global_path, local_path


    def draw_path(self, path, map_height, map_width, resolution):
        self.local_map = np.zeros((map_height, map_width))

        # transformation = torch.Tensor([[-1/resolution, 0, map_height-1], 
        #                                [0, -1/resolution, int((map_width+1)/2)-1],
        #                                [0, 0, 1]])
        transformation = torch.Tensor([[0, -1/resolution, int((map_width+1)/2)-1], 
                                       [1/resolution, 0, 0],
                                       [0, 0, 1]])
        homo_path = torch.transpose(torch.cat([path[:,:2], torch.ones(path.shape[0], 1)], 1), 0, 1)
        pixel_path = torch.matmul(transformation, homo_path)
        pixel_path = torch.transpose(pixel_path[:2, :], 0, 1).int()

        print(f"Pixel path: {pixel_path.shape}")
        print(pixel_path)
        print(f"Yaw: {path[:,2].view(-1,1)}")
        print(f"Yaw shape: {path[:,2].view(-1,1).shape}")

        pixel_path = torch.cat([pixel_path, path[:,2].view(-1,1)], 1)
        print("pixel_path")
        print(pixel_path)
        print(f"pixel_path shape: {pixel_path.shape}")

        pixel_x = pixel_path[:,0].numpy()
        pixel_y = pixel_path[:,1].numpy()

        cond1 = np.argwhere(pixel_x >= 0)
        cond2 = np.argwhere(pixel_x < map_height)
        cond3 = np.argwhere(pixel_y >= 0)
        cond4 = np.argwhere(pixel_y < map_width)

        ind_in_frame = reduce(np.intersect1d, [cond1, cond2, cond3, cond4])

        valid_path = pixel_path[ind_in_frame, :]

        self.local_map[valid_path[:,0].int(), valid_path[:,1].int()] = 2
        # print(f"valid path: {valid_path.shape}")
        # print(valid_path)

        # plt.imshow(self.local_map, origin='lower')
        # plt.xlabel("X axis")
        # plt.ylabel("Y axis")
        # plt.show()

        return valid_path


if __name__ == "__main__":
    # Load in trajectory
    base_dir = '/home/mateo/Data/SARA/PyTorchDatasets'
    device = "gpu"

    tl = TrajLoader(base_dir, device)
    traj = tl.getTrajectory()

    traj = traj[0]

    slicer = lambda x: x[180:280]
    # slicer = lambda x: x[400:500]
    short_traj = dict_map(traj, slicer)

    map_height = 200
    map_width = 200
    resolution = 0.05
    path_transform = PathTransform(short_traj, map_height, map_width, resolution)
