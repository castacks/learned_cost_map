from matplotlib import pyplot as plt 
import numpy as np
np.set_printoptions(suppress=True)
from PIL import Image

maps = np.load('rgb_map_list.npy')
odoms = np.load('local_pose_list.npy')
times = np.load('ts_list.npy')

maps_size = maps[0].shape

print(maps.shape, odoms.shape)
plt.subplot(131)
plt.imshow(maps[0], origin='lower')
plt.scatter(odoms[:,0], odoms[:,1])

for odom in odoms:
    pixel_dx = np.cos(odom[2])*20
    pixel_dy = np.sin(odom[2])*20

    plt.arrow(odom[0]-0.5*pixel_dx, odom[1]-0.5*pixel_dy, pixel_dx, pixel_dy, color="red", head_width=10)
plt.subplot(132)
plt.imshow(maps[2])
plt.subplot(133)
new_x = int(odoms[1,0])
new_y = int(odoms[0,1]) - int(odoms[1,1]) 

print(odoms[0,:])
print(odoms[1,:])

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

def merge_images(file1, file2):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.fromarray(np.uint8(file1*255)).convert('RGB')
    image2 = Image.fromarray(np.uint8(file2*255)).convert('RGB')

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(new_x, new_y))
    return result
# Left image
# height, width, _ = maps[0].shape
# left_corners = np.array([0,0,1], [0,height,1])

# corner1 = 0#int(new_x-int(maps_size[0]/2))
# corner2 = int(new_x+int(maps_size[0]/2))
# corner3 = 0# int(new_y-int(maps_size[1]/2))
# corner4 = int(new_y+maps_size[1]/2)
# print(corner1, corner2, corner3, corner4)
# maps[0][corner1:corner2, corner3:corner4] = maps[1,:165, :198]
# plt.imshow(maps[0])
# plt.title("Stitched")
# plt.subplot(133)
# plt.imshow(add_border(maps[0],3))
merged = merge_images(add_border(maps[0],3), add_border(maps[1],3))
plt.imshow(merged)
plt.show()