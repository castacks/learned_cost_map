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
    def __init__(self, input_channels, embedding_size, output_size, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=embedding_size, bias=True)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=1, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+512, out_features=output_size)
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
    def __init__(self, input_channels, ff_size, embedding_size, output_size, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=embedding_size, bias=True)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=ff_size*2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+512, out_features=output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data["patches"]
        vel = input_data["fourier_vels"]
        import pdb;pdb.set_trace()
        processed_maps = self.model(x)
        processed_vel  = self.vel_mlp(vel)
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
    def __init__(self, input_channels, ff_size, embedding_size, output_size, pretrained=False):
        super().__init__()
        self.model = models.efficientnet_b0(pretrained)
        self.model.features[0][0] = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=embedding_size, bias=True)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=ff_size*2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+512, out_features=output_size)
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
    def __init__(self, input_channels, ff_size, embedding_size, output_size, pretrained=False):
        super().__init__()
        self.large_model = models.resnet18(pretrained)
        self.large_model.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.large_model.fc = nn.Linear(in_features=128, out_features=embedding_size, bias=True)

        self.model = nn.Sequential(self.large_model.conv1, self.large_model.bn1, self.large_model.relu, self.large_model.maxpool, self.large_model.layer1, self.large_model.layer2, self.large_model.avgpool)#, self.large_model.fc)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=ff_size*2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+512, out_features=output_size)
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
    def __init__(self, input_channels, ff_size, embedding_size, output_size, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained)
        self.model.fc = nn.Linear(in_features=512, out_features=embedding_size, bias=True)

        self.vel_mlp = nn.Sequential(
            nn.Linear(in_features=ff_size*2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(in_features=embedding_size+512, out_features=output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        import pdb;pdb.set_trace()
        x = input_data["patches"][:,0:3,:,:]
        vel = input_data["fourier_vels"]
        processed_maps = self.model(x)
        processed_vel  = self.vel_mlp(vel)
        combined_features = torch.cat([processed_maps, processed_vel], dim=1)
        output = self.output_mlp(combined_features)
        output = self.sigmoid(output)
        return output

if __name__ == "__main__":
    model = CostModelEfficientNet(8, 1)
    print(model)
