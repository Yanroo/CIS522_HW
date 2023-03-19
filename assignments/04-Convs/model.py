import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    The model class.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        cout_dim1 = 10
        kernel_size = 3
        self.pool_size = 5
        after_kernel = (32 - 2 * (kernel_size // 2)) // self.pool_size
        # out_dim2 = 32
        # fout_dim1 = 16
        # fout_dim2 = 16
        # self.conv1_dw = nn.Conv2d(num_channels, num_channels, 3, 1, padding="same", groups=num_channels)
        # self.conv1_pw = nn.Conv2d(num_channels, out_dim1, 1, 1)

        self.conv1 = nn.Conv2d(num_channels, cout_dim1, kernel_size, 1)
        # self.conv2 = nn.Conv2d(out_dim1, out_dim2, 1, 1, padding="same")#
        self.fc1 = nn.Linear(cout_dim1 * after_kernel * after_kernel, num_classes)
        # self.fc2 = nn.Linear(fout_dim1, num_classes)
        # self.fc1 = nn.Linear(960, fout_dim1)
        # self.bn1 = nn.BatchNorm1d(fout_dim1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        x = self.conv1(x)
        # x = self.conv1_dw(x)
        # x = self.conv1_pw(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # print("shape1: ", x.shape)
        x = F.max_pool2d(x, self.pool_size)
        # print("shape2: ", x.shape)
        x = torch.flatten(x, 1)
        # x = F.max_pool1d(x, 2)
        x = self.fc1(x)
        # x = self.bn1(x)
        # x = F.relu(x)

        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)

        return x
