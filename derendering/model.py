import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import ipdb


class DeRendering(nn.Module):
    def __init__(self, num_objects=5):
        super().__init__()

        # Number of objects
        self.num_objects = num_objects

        # CNN
        self.cnn = models.resnet18(pretrained=True)

        # Presence
        self.mlp_presence = nn.Sequential(nn.Linear(512, num_objects))

        # Properties
        self.D = 16  # times 7x7
        self.last_cnn = nn.Sequential(nn.Conv2d(512, self.D * self.num_objects,
                                                kernel_size=1, stride=1, padding=0),
                                      nn.ReLU())
        self.mlp_pose = nn.Sequential(nn.Linear(self.D * 7 * 7, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 3 + 4),
                                      )

    def forward(self, x):
        # cnn
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)
        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # presence
        avgpool = torch.mean(x, [2, 3])
        presence = self.mlp_presence(avgpool)

        # properties
        x = self.last_cnn(x)
        x = x.view(-1, self.num_objects, self.D * 7 * 7)
        pose = self.mlp_pose(x)
        pose_3d = pose[:, :, :3]
        pose_2d = torch.cat([
            F.sigmoid(pose[:, :, 3:5]),
            F.sigmoid(pose[:, :, 3:5] + pose[:, :, 5:])
        ], -1)

        return presence, pose_3d, pose_2d
