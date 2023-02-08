import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Callable


class MLP(nn.Module):
    """
    A multi-layer perceptron.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        # hidden_dims: list, #for parameter tuning
        hidden_count: int = 1,
        activation: Callable = nn.ReLU,
        initializer: Callable = nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.actv = activation()
        self.dropout = nn.Dropout(p=0.5)

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.fc = nn.Linear(in_features=512, out_features=num_classes)
        # self.conv1 = nn.Conv2d(
        #     in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        # )
        # self.conv2 = nn.Conv2d(
        #     in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        # )
        # self.fc = nn.Linear(in_features=3136, out_features=10)

        for _ in range(hidden_count):
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        # for i in range(len(hidden_dims)):
        #     hidden_size = hidden_dims[i]
        #     self.layers.append(nn.Linear(input_size, hidden_size))
        #     input_size = hidden_size

        self.out = nn.Linear(input_size, num_classes)
        for layer in self.layers:
            initializer(layer.weight)
        initializer(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = self.actv(layer(x))
            # x = self.dropout(x)
        x = self.out(x)

        # x = x.reshape(-1, 1, 28, 28)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size(0), -1)
        # x = self.fc(x)
        # x = x.reshape(-1, 1, 28, 28)
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x
