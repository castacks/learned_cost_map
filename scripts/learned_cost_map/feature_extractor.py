# Color feature extractor
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt
import cv2

class FeatureExtractor:
    def __init__(self, image_tensor):
        self.image_tensor = image_tensor
        self.input_image_size = 64
        # self.patch_size = 64
        self.batch_size = self.image_tensor.shape[0]
        self.num_bins = 4
        self.num_channels = 3
        self.num_patches = 4 # Implicit using FiveCrop transform and ignoring last value
    def get_features(self):

        # testTransform = T.Compose([
        #     T.ToTensor(),
        #     T.Resize(size=self.input_image_size),
        #     T.FiveCrop(size=(int(self.input_image_size/2), int(self.input_image_size/2))),
        # ])

        # patches = testTransform(self.image_tensor)
        # print(f"Patches size: {patches[0].shape}")
        # # patches = torch.stack()




        getCorners = T.Compose([
            # T.ToTensor(),
            T.Resize(size=self.input_image_size),
            T.FiveCrop(size=(int(self.input_image_size/2), int(self.input_image_size/2))),
            T.Lambda(lambda crops: torch.stack([(crop) for crop in crops[:-1]], dim=1))
        ])

        patches = getCorners(self.image_tensor) # Ignore last three channels since they correspond to the center crop and we only care about corners

        per_image_hist = []
        for batch in range(patches.shape[0]):
            per_patch_hist = []
            for j in range(patches.shape[1]):
                per_channel_hist = []
                for c in range(self.num_channels):
                    channel_patch_hist = torch.histc(patches[batch, j, c, :, :], bins=self.num_bins, min=0, max=1)
                    per_channel_hist.append(channel_patch_hist)
                per_channel_hist = torch.cat(per_channel_hist, dim=0)
                per_patch_hist.append(per_channel_hist)
            per_patch_hist = torch.cat(per_patch_hist, dim=0)
            per_image_hist.append(per_patch_hist)
        # print(f"length per_image_hist: {len(per_image_hist)}")
        # print(f"Size of elements of list above: {per_image_hist[0].shape}")
        histogram = torch.stack(per_image_hist, dim=0)
        histogram = histogram / torch.sum(histogram, dim=1).view(-1,1)
        # print(histogram.shape)
        # print(torch.sum(histogram))

        return histogram
        

        #         per_channel_hist = [torch.histc(patches[batch, j, c, :, :]) for c in range(self.num_channels)]
        #         per_channel_hist = torch.cat(per_channel_hist, dim=0)
        #         image_hist.append(torch.histc(patches[batch,j,:,:], bins=self.num_bins, ))

        # hist = [torch.histc(patches[i], bins=4, min=0, max=1) for i in range(patches.shape[0])]
        # print(len(hist))
        # hist = torch.cat(hist, dim=0)
        # hist = hist/torch.sum(hist)

        # print(hist.shape)
        # print(torch.sum(hist))




if __name__=="__main__":
    rgb_map = np.load('/home/mateo/rgb_map.npy')

    crops = torch.load('/home/mateo/SARA/src/sara_ws/src/traversability_cost/scripts/crops.pt')
    crops = crops[:, :3, :, :]
    print(f"Crops shape: {crops.shape}")

    fe = FeatureExtractor(crops)

    fe.transform()