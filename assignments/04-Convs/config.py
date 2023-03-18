from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize


class CONFIG:
    batch_size = 128
    num_epochs = 8

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=3e-3)
    
    #torch.optim.SGD(model.parameters(), lr=8e-2, momentum=0.9)
    

    transforms = Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
