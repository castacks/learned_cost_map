from lib2to3.pytree import convert
from turtle import left, right
from matplotlib import pyplot as plt 
import numpy as np
np.set_printoptions(suppress=True)
from PIL import Image

maps = np.load('rgb_map_list.npy')
local_odoms = np.load('local_pose_list.npy')
times = np.load('ts_list.npy')

def convert_local_to_pixel(local_path):
    """Convert local_path to pixel space.
    Local_path: N x 3 numpy array (x, y, theta)
    Output: Nx3 Numpy array
    """
    x = local_path[:,0]
    y = local_path[:,1]
    theta = local_path[:,2]
    pixel_x = (x/resolution).astype(int)
    pixel_y = ((y - origin[1])/resolution).astype(int)
    pixel_xyt = np.vstack((pixel_x, pixel_y, theta)).T
    return pixel_xyt

def convert_local_to_grid(local_path, origin):
    """Convert local_path to pixel space.
    Local_path: N x 2 numpy array (x, y)
    Output: Nx2 Numpy array
    """
    x = local_path[:,0]
    y = local_path[:,1]
    pixel_y = ((x - origin[0])/resolution).astype(int)
    pixel_x = ((y - origin[1])/resolution).astype(int)
    pixel_xyt = np.vstack((pixel_x, pixel_y)).T
    return pixel_xyt

def convert_grid_to_local(grid_x, grid_y, odom_x, odom_y, odom_theta):

    # grid to meter
    y = grid_x * resolution + origin[1]
    x = (grid_y * resolution)
    xy = np.vstack((x,y))
    print("xy shape", xy.shape)
    # apply rotation 
    rotated_x = np.cos(odom_theta) * x - np.sin(odom_theta) * y
    rotated_y = np.sin(odom_theta) * x + np.cos(odom_theta) * y

    # apply translation 
    translated_x = rotated_x + odom_x
    translated_y = rotated_y + odom_y


    xyt = np.vstack((translated_x, translated_y)).T
    return xyt

def add_border(image, border_width):
    # top border
    image[0:border_width,:] = [1,0,0]
    # bottom
    image[image.shape[0]-border_width:,:] = [1,0,0]

    # left
    image[:,:border_width] = [1,0,0]

    # right
    image[:,image.shape[1]-border_width:] = [1,0,0]
    return image

origin = [0,-5]
resolution = 0.05
height_m = 10
width_m = 10
height = height_m/resolution
width = width_m/resolution
xmin = origin[0]
xmax = origin[0] + width
ymin = origin[1]
ymax = origin[1] + height

print("NUM MAPS: ", maps.shape)
# for ind in range(maps.shape[0])[::10]:
combined_coord = np.zeros((0,2))
combined_grid_color = np.zeros((0,3))
for ind in [0,50, 100, 125, 150, 175, 200, 250, 300]:
# ind = 1
    map = add_border(maps[ind],0)
    odom = local_odoms[ind]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_x = grid_x.flatten().astype(int)#[::] 
    grid_y = grid_y.flatten().astype(int)#[::2]
    grid_color = map[grid_x, grid_y]
    local_coord = convert_grid_to_local(grid_x, grid_y, odom[0], odom[1], odom[2])
    # print(local_coord.shape)
    # plt.scatter(local_coord[:,0], local_coord[:,1], c=grid_color, alpha=1) # plot local map
    combined_coord = np.vstack((combined_coord, local_coord))
    combined_grid_color = np.vstack((combined_grid_color, grid_color))
    # print(combined_coord.shape)
    # plt.scatter(local_coord[:,0], local_coord[:,1], c=grid_color, alpha=1)
    # print(convert_grid_to_local(100,0, odom[0], odom[1], 0))

    # plt.scatter(local_odoms[:,0], local_odoms[:,1]) # all odometries
    # plt.scatter(odom[0], odom[1], c='r', alpha=1) # current odometry
    # pixel_dx = np.cos(odom[2])*1
    # pixel_dy = np.sin(odom[2])*1
    # plt.arrow(odom[0]-0.5*pixel_dx, odom[1]-0.5*pixel_dy, pixel_dx, pixel_dy, color="blue", head_width=0.2)


# Visualize: Plot local map (metric)
# plt.xlabel("x")
# plt.ylabel('y')
# plt.axis('equal')
# plt.scatter(combined_coord[:,0], combined_coord[:,1], c=combined_grid_color, alpha=1) 
# plt.show()

# Convert combined metric map to grid. Find basic metadata for combined grid map.  

leftmost_x_m = np.min(combined_coord[:,0])
rightmost_x_m = np.max(combined_coord[:,0])
topmost_x_m = np.max(combined_coord[:,1])
botmost_x_m = np.min(combined_coord[:,1])

orig_leftmost_x_m = 0
orig_rightmost_x_m = width_m + origin[0] # or height
orig_topmost_x_m = height_m + origin[1] # or width
orig_botmost_x_m = 0

print(leftmost_x_m, rightmost_x_m, topmost_x_m, botmost_x_m)
combined_width = rightmost_x_m - leftmost_x_m
combined_width_grid = np.ceil(combined_width/resolution).astype(int)
combined_height = topmost_x_m - botmost_x_m
combined_height_grid = np.ceil(combined_height/resolution).astype(int)
# print(combined_width_pix, combined_height_pix)

origin_shift_x_m = leftmost_x_m - orig_leftmost_x_m
origin_shift_y_m = topmost_x_m - orig_topmost_x_m
print("corners", leftmost_x_m, rightmost_x_m, topmost_x_m, botmost_x_m)
print("Origin shift grid", origin_shift_x_m, origin_shift_y_m)

new_map_local = np.zeros((combined_height_grid, combined_width_grid, 3))
grid_x, grid_y = np.meshgrid(np.arange(combined_height_grid), np.arange(combined_width_grid))
grid_x = grid_x.flatten().astype(int)
grid_y = grid_y.flatten().astype(int)

# for every point in new local map, we want an associated color (if there's nothing nearby, assign 0)
new_origin = [leftmost_x_m, botmost_x_m] 
new_grid_coords = convert_local_to_grid(combined_coord, new_origin) # TODO: this should be replaced to new grid metadata
print("new map size", new_map_local.shape)
print("max new grid", np.max(new_grid_coords[:,0]), np.max(new_grid_coords[:,1]))
print("combined shape", combined_grid_color.shape, combined_coord.shape)
new_map_local[new_grid_coords[:,0], new_grid_coords[:,1]] = combined_grid_color

plt.imshow(new_map_local)

# overlay trajectory
new_odom_grid = convert_local_to_grid(local_odoms, new_origin)
plt.scatter(new_odom_grid[:,1], new_odom_grid[:,0]) # all odometries

plt.show()

# maps_size = maps[0].shape

# print(maps.shape, odoms.shape)
# plt.subplot(131)
# plt.imshow(maps[0], origin='lower')
# plt.scatter(odoms[:,0], odoms[:,1])

# for odom in odoms:
#     pixel_dx = np.cos(odom[2])*20
#     pixel_dy = np.sin(odom[2])*20

#     plt.arrow(odom[0]-0.5*pixel_dx, odom[1]-0.5*pixel_dy, pixel_dx, pixel_dy, color="red", head_width=10)
# plt.subplot(132)
# plt.imshow(maps[2])
# plt.subplot(133)
# new_x = int(odoms[1,0])
# new_y = int(odoms[0,1]) - int(odoms[1,1]) 

# print(odoms[0,:])
# print(odoms[1,:])


