import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, depth: int):
        """
        Initializes the DnCNN with the specified depth.

        Parameters
        ----------
        depth : int
            The number of intermediate convolutional layers.
        """
        super(DnCNN, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        ]
        
        for _ in range(depth):
            layers.extend([
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_features=64, eps=0.0001, momentum=0.95),
                nn.ReLU()
            ])
        
        layers.append(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False)
        )
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the DnCNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after processing through DnCNN.
        """
        return self.model(x)
