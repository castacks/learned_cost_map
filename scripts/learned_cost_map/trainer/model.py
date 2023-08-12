import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

class CostModel(nn.Module):
    def __init__(self, input_channels, output_size, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=output_size, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data["patches"]
        output = self.model(x)
        output = self.sigmoid(output)
        return output


class CostVelModel(nn.Module):
    def __init__(self, input_channels=8, embedding_size=512, mlp_size=512, output_size=1, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=embedding_size, bias=True)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=1, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+mlp_size, out_features=output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["vels"]
        processed_maps = self.model(x)
        processed_vel  = self.vel_mlp(vel)
        combined_features = torch.cat([processed_maps, processed_vel], dim=1)
        output = self.output_mlp(combined_features)
        output = self.sigmoid(output)
        return output


class CostFourierVelModel(nn.Module):
    def __init__(self, input_channels=8, ff_size=16, embedding_size=512, mlp_size=512, output_size=1, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=embedding_size, bias=True)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=ff_size*2, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+mlp_size, out_features=output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["fourier_vels"]
        # import pdb;pdb.set_trace()
        processed_maps = self.model(x)
        processed_vel  = self.vel_mlp(vel)
        if len(processed_maps.shape) < 2:
            processed_maps = processed_maps.view(1, -1)
        if len(processed_vel.shape) < 2:
            processed_vel = processed_vel.view(1, -1)
        combined_features = torch.cat([processed_maps, processed_vel], dim=1)
        output = self.output_mlp(combined_features)
        output = self.sigmoid(output)
        return output


class CostModelEfficientNet(nn.Module):
    def __init__(self, input_channels, output_size, pretrained=False):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained)
        self.model.features[0][0] = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=output_size, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data["patches"]
        output = self.model(x)
        output = self.sigmoid(output)
        return output

class CostFourierVelModelEfficientNet(nn.Module):
    def __init__(self, input_channels=8, ff_size=16, embedding_size=512, mlp_size=512, output_size=1, pretrained=False): 
        super().__init__()
        self.model = models.efficientnet_b0(pretrained)
        self.model.features[0][0] = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=embedding_size, bias=True)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=ff_size*2, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+mlp_size, out_features=output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["fourier_vels"]
        processed_maps = self.model(x)
        processed_vel  = self.vel_mlp(vel)
        combined_features = torch.cat([processed_maps, processed_vel], dim=1)
        output = self.output_mlp(combined_features)
        output = self.sigmoid(output)
        return output

class CostFourierVelModelSmall(nn.Module):
    def __init__(self, input_channels=8, ff_size=16, embedding_size=512, mlp_size=512, output_size=1, pretrained=False):
        super().__init__()
        self.large_model = models.resnet18(pretrained)
        self.large_model.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.large_model.fc = nn.Linear(in_features=128, out_features=embedding_size, bias=True)

        self.model = nn.Sequential(self.large_model.conv1, self.large_model.bn1, self.large_model.relu, self.large_model.maxpool, self.large_model.layer1, self.large_model.layer2, self.large_model.avgpool)#, self.large_model.fc)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=ff_size*2, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+mlp_size, out_features=output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["fourier_vels"]
        processed_maps = self.large_model.fc(self.model(x).squeeze())
        processed_vel  = self.vel_mlp(vel)
        combined_features = torch.cat([processed_maps, processed_vel], dim=1)
        output = self.output_mlp(combined_features)
        output = self.sigmoid(output)
        return output

class CostFourierVelModelRGB(nn.Module):
    def __init__(self, input_channels=8, ff_size=16, embedding_size=512, mlp_size=512, output_size=1, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained)
        self.model.fc = nn.Linear(in_features=512, out_features=embedding_size, bias=True)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=ff_size*2, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+mlp_size, out_features=output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        # import pdb;pdb.set_trace()
        x = input_data["patches"][:,0:3,:,:]
        vel = input_data["fourier_vels"]
        processed_maps = self.model(x)
        processed_vel  = self.vel_mlp(vel)
        combined_features = torch.cat([processed_maps, processed_vel], dim=1)
        output = self.output_mlp(combined_features)
        output = self.sigmoid(output)
        return output

class EnsembleHead(nn.Module):
    def __init__(self, mlp_size=32, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=output_size)
        )

    def forward(self, input_data):
        out = self.model(input_data)

        return out

class EnsembleCostFourierVelModel(nn.Module):
    def __init__(self, input_channels=8, ff_size=16, embedding_size=512, mlp_size=512, output_size=1, num_heads=32, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=embedding_size, bias=True)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=ff_size*2, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU(),
            nn.Linear(in_features=mlp_size, out_features=mlp_size),
            nn.ReLU()
        )

        self.output_ensemble = nn.ModuleList(EnsembleHead(mlp_size=embedding_size+mlp_size, output_size=output_size) for k in range(num_heads))

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+mlp_size, out_features=output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["fourier_vels"]
        # import pdb;pdb.set_trace()
        processed_maps = self.model(x)
        processed_vel  = self.vel_mlp(vel)
        if len(processed_maps.shape) < 2:
            processed_maps = processed_maps.view(1, -1)
        if len(processed_vel.shape) < 2:
            processed_vel = processed_vel.view(1, -1)
        combined_features = torch.cat([processed_maps, processed_vel], dim=1)

        outputs = []
        for i, head in enumerate(self.output_ensemble):
            output = head(combined_features)
            output = self.sigmoid(output)
            outputs.append(output)

        # output = self.output_mlp(combined_features)
        # output = self.sigmoid(output)
        return outputs

class BaselineGeometricModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=2, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["vels"]
        nn_input = self.patches_vel_to_input(x, vel)
        output = self.model(nn_input)
        return output

    def patches_vel_to_input(self, patches, vel):
        patches_features = torch.mean(patches[:,6,:,:], dim=[-1, -2]).view(-1, 1)
        nn_input = torch.cat([patches_features, vel], dim=1)#.squeeze()
        # import pdb;pdb.set_trace()
        return nn_input

class BaselineVisualGeometricModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=5, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["vels"]
        nn_input = self.patches_vel_to_input(x, vel)
        output = self.model(nn_input)
        return output

    def patches_vel_to_input(self, patches, vel):
        patches_std_features = torch.mean(patches[:,6,:,:], dim=[-1, -2]).view(-1, 1)
        patches_red_features = torch.mean(patches[:,0,:,:], dim=[-1, -2]).view(-1, 1)
        patches_green_features = torch.mean(patches[:,1,:,:], dim=[-1, -2]).view(-1, 1)
        patches_blue_features = torch.mean(patches[:,2,:,:], dim=[-1, -2]).view(-1, 1)
        nn_input = torch.cat([patches_std_features, patches_red_features, patches_green_features, patches_blue_features, vel], dim=1)#.squeeze()
        # import pdb;pdb.set_trace()
        return nn_input

class BaselineGeometricLargeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=4097, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["vels"]
        nn_input = self.patches_vel_to_input(x, vel)
        output = self.model(nn_input)
        return output

    def patches_vel_to_input(self, patches, vel):
        patches_std_features = torch.flatten(patches[:,6,:,:], start_dim=1)
        nn_input = torch.cat([patches_std_features, vel], dim=1)#.squeeze()
        # import pdb;pdb.set_trace()
        return nn_input


class BaselineVisualGeometricLargeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=16385, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["vels"]
        nn_input = self.patches_vel_to_input(x, vel)
        output = self.model(nn_input)
        return output

    def patches_vel_to_input(self, patches, vel):
        patches_std_features = torch.flatten(patches[:,6,:,:], start_dim=1)
        patches_red_features = torch.flatten(patches[:,0,:,:], start_dim=1)
        patches_green_features = torch.flatten(patches[:,1,:,:], start_dim=1)
        patches_blue_features = torch.flatten(patches[:,2,:,:], start_dim=1)
        nn_input = torch.cat([patches_std_features, patches_red_features, patches_green_features, patches_blue_features, vel], dim=1)#.squeeze()
        # import pdb;pdb.set_trace()
        return nn_input

if __name__ == "__main__":
    model = CostModelEfficientNet(8, 1)
    print(model)
