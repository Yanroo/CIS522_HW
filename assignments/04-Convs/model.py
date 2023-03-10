import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    The model class.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        out_dim1 = 32
        # out_dim2 = 64
        self.conv1 = nn.Conv2d(num_channels, out_dim1, 3, 1, padding="same")
        # self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")#
        self.fc1 = nn.Linear(out_dim1 * 32 * 32 // 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # print("shape1: ", x.shape)
        x = F.max_pool2d(x, 2)
        # print("shape2: ", x.shape)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
