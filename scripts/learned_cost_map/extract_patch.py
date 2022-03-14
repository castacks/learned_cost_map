import torch 
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt


class PatchTransform:
    '''Defines PyTorch transform to obtain a patch on an image facing a specific direction.

    Args:
        - position:
            Tuple/list of (x, y) coordinates in the image, where the origin is the top left of the image, +x points down, and +y points to the right. 
        - yaw:
            Float. Angle (in radians) of rotation in FLU. The yaw is with respect to the following frame: +x points up, +y points to the left, and positive rotations are counterclockwise 
        - crop_size:
            Tuple/list of size of the patch we want to obtain in pixel space. 
        - output_size:
            Tuple/list of size of the output image. This is an upsampled/downsampled version of the crop_size
        - on_top (Optional):
            Boolean. If True, it generates the patch with the input position at the center of the patch. This is what we would use if we want to obtain the patch directly under the robot. If False, it generates the patch directly in front of the robot. This is the patch that the robot would see in front of it.
    '''
    def __init__(self, position, yaw, crop_size, output_size, on_top=True):
        self.position = position
        self.yaw = yaw
        self.crop_size = crop_size
        self.output_size = output_size
        self.on_top = on_top

        self.yaw_deg = self.yaw*180/np.pi

    def __call__(self, x):
        '''Callable method for transform.
        
        Args:
            - x:
                Torch image of size NxCxHxW   
        '''
        width = x.shape[-2]
        height = x.shape[-1]

        # First, translate to make the position of the robot the center of the image
        rotated_x = np.cos(self.yaw) * 125 # TODO: comment
        rotated_y = np.sin(self.yaw) * 125
        translation = [round(height/2)-self.position[0]-rotated_x, round(width/2)-self.position[1]-rotated_y]
        # print("Translation", translation)
        # print("Width, height", width, height)

        output = TF.affine(x, angle=0.0, translate=translation, scale=1.0, shear=0.0)

        # Second, rotate the image around x axis to the right yaw, where x axis is a vector pointing up
        angle = self.yaw_deg # TODO check
        # angle = self.yaw_deg

        output = TF.rotate(output, angle=angle, expand=False)

        if not self.on_top:
            # Third, translate again to put the robot at the bottom of the frame: we don't want the patch to be centered around the robot, but rather for it to be what the robot sees from that location.
            pov_translation = [0,self.crop_size[0]/2] 
            output = TF.affine(output, angle=0.0, translate=pov_translation, scale=1.0, shear=0.0)

        # Now, get a center crop of the right patch size of the image to be analyzed
        output = TF.center_crop(output, self.crop_size)

        # Finally, resize the crop to fit the input dimensions of the neural network
        output = TF.resize(output, size=self.output_size)

        return output


if __name__=="__main__":
    rgb_map = np.swapaxes(np.load('/home/mateo/rgb_map.npy'), 0, 1)
    print(f"RGB map shape: {rgb_map.shape}")
    # rgb_map = np.copy(np.flipud(np.fliplr(rgb_map)))
    print(f"RGB map shape: {rgb_map.shape}")
    height_map = np.swapaxes(np.load('/home/mateo/height_map.npy'), 0, 1)
    image = np.load('/home/mateo/image.npy')

    # print("Original rgb_map")
    # plt.imshow(rgb_map, origin="lower")
    # plt.xlabel("X axis")
    # plt.ylabel("Y axis")
    # plt.show()
    yaw = np.pi/4
    # position = [0, 0]
    position = [0, 100]
    # position = [480, 320]
    crop_size = [100,100]
    output_size = [100, 100]

    getPatch = T.Compose([
        T.ToTensor(),
        PatchTransform(position, yaw, crop_size, output_size, on_top=True),
    ])

    patch = getPatch(rgb_map)

    patch = patch.permute(1,2,0).numpy()

    print(f"Shape of patch: {patch.shape}")



    imgs = [rgb_map, patch]
    # https://stackoverflow.com/questions/41210823/using-plt-imshow-to-display-multiple-images
    def show_images(images):
        n: int = len(images)
        f = plt.figure()
        for i in range(n):
            # Debug, plot figure
            f.add_subplot(1, n, i + 1)
            plt.imshow(images[i], origin="lower")
            plt.xlabel("X axis")
            plt.ylabel("Y axis")

        plt.show(block=True)

    show_images(imgs)
