import torch.nn as nn

def DnCNN(depth: int) -> nn.Sequential:
    """
    Generates a DnCNN with the specified depth.

    Parameters
    ----------
    depth : int
        The number of intermediate convolutional layers.

    Returns
    -------
    nn.Sequential
        A sequential model containing the layers defined by DnCNN.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=True),
        nn.ReLU(),
        *[
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_features=64, eps=0.0001, momentum=0.95),
                nn.ReLU()
            )
            for _ in range(depth)
        ],
        nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False),
        nn.Flatten()
    )
