from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 64
    num_epochs = 14
    initial_learning_rate = 0.002
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "step_size": 0.2 * num_epochs * 50000 // batch_size,
        "verbose": True,
        "gamma": 0.3,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )
    # ] = lambda model: torch.optim.SGD(
    #     model.parameters(),
    #     lr=CONFIG.initial_learning_rate,
    #     momentum=0.9,
    #     weight_decay=CONFIG.initial_weight_decay,
    # )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )
