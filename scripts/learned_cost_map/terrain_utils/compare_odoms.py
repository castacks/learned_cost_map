## Compare odometries
import numpy as np
import matplotlib.pyplot as plt
import torch
from learned_cost_map.terrain_utils.terrain_map_tartandrive import get_local_path

odom = torch.from_numpy(np.load("/home/mateo/Data/SARA/TartanDriveCost/Trajectories/000009/odom/odometry.npy"))
tartanvo_odom = torch.from_numpy(np.load("/home/mateo/Data/SARA/TartanDriveCost/Trajectories/000009/tartanvo_odom/poses.npy"))

local_path = get_local_path(odom)
local_path_tartanvo = get_local_path(tartanvo_odom)

print(local_path)
print(local_path_tartanvo)

plt.scatter(local_path[:,0], local_path[:,1], label="Odom")
plt.scatter(local_path_tartanvo[:,1], local_path_tartanvo[:,0], label="TartanVO Odom")
plt.legend()
plt.show()