import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Union, Optional

def init_weights(mu: float, size: int) -> torch.Tensor:
    """
    Initialize weights for the convolutional layers in a gradient descent form.

    Parameters
    ----------
    mu : float
        The step size for gradient descent.
    size : int
        The size of the weights to be initialized.

    Returns
    -------
    torch.Tensor
        Initialized weight tensor.
    """
    t = torch.zeros(1, size, 1, 1)
    t[0][0] = 1
    t[0][1] = mu

    return t

class Block(nn.Module):
    """
    A Block within the EndToEnd model consisting of convolutional layers and gradient step adjustments.

    Attributes
    ----------
    model : nn.Sequential
        A sequential container of convolutional and activation layers.
    matrix : torch.Tensor
        The sensing matrix used for transformation.
    """

    def __init__(self, A: np.ndarray, depth: int, device: torch.device):
        """
        Initializes the Block with the given parameters.

        Parameters
        ----------
        A : np.ndarray
            The sensing matrix.
        depth : int
            Number of convolutional layers in the block.
        device : torch.device
            The device on which to perform computations (CPU or GPU).
        """
        super().__init__()

        self.model: nn.Sequential = nn.Sequential(
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
        self.matrix: torch.Tensor = torch.from_numpy(A).float().to(device)   
        self.device = device

    def forward(self, y: torch.Tensor, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the block.

        Parameters
        ----------
        y : torch.Tensor
            The input tensor, typically the observed data.
        x : torch.Tensor
            The initial tensor to be updated in this block.
        memory : Optional[torch.Tensor]
            Memory tensor to store intermediate results across layers.

        Returns
        -------
        torch.Tensor
            The updated tensor after processing through the block.
        torch.Tensor
            The new memory tensor after processing.
        """
        if memory is not None:
            # Memeory from the previous block
            d, c, h, w = memory.size()
        else:
            # Initialize memory and dimensions if not provided
            d, _, h, w = x.size()
            memory = torch.zeros(d, 2, h, w, device=self.device)   
            memory[:, 0, :, :] = torch.zeros(d, h, w, device=self.device)  # Initialize first memory slot with zeros
            memory[:, 1, :, :] = torch.reshape((self.matrix.t() @ y.t()).t(), (d, h, w))  # Last memory term is A^T y
            c = 2

        # Projection Step
        x_new = torch.reshape(x, (d, h * w)) + self.model(x)

        # Retrieve the newest gradient
        residual = y - (self.matrix @ x_new.t()).t()
        x_tilde = (self.matrix.t() @ residual.t()).t()

        # Update memory
        new_memory = torch.zeros(d, c + 1, h, w, device=self.device)
        new_memory[:, 1:, :, :] = memory
        new_memory[:, 1, :, :] = torch.reshape(x_tilde, (d, h, w))
        new_memory[:, 0, :, :] = torch.reshape(x_new, (d, h, w))

        return torch.reshape(x_new, (d, 1, h, w)), new_memory

class MemoryNetwork(nn.Module):
    """
    MemoryNetwork network model for projected gradient descent, processing input through multiple projections.

    Attributes
    ----------
    numProjections : int
        Number of projection steps (layers) in the network.
    mu_steps : List[float]
        List of step sizes for each projection step.
    mu_init : torch.nn.Parameter
        Initial step size parameter.
    matrix : torch.Tensor
        The sensing matrix used in the model.
    net : nn.ModuleList
        List containing all the Block layers and convolutional layers for projections.
    device : torch.device
        The device on which to perform computations (CPU or GPU).
    """

    def __init__(self, A: np.ndarray, mu: List[float], numProjections: int, depth: int, device: torch.device):
        """
        Initializes the EndToEnd model with the given parameters.

        Parameters
        ----------
        A : np.ndarray
            The sensing matrix.
        mu : List[float]
            A list of step sizes for each projection step.
        numProjections : int
            Number of projections (layers) in the network.
        depth : int
            Number of convolutional layers within each Block.
        device : torch.device
            The device on which to perform computations (CPU or GPU).
        """
        super().__init__()

        self.numProjections: int = numProjections
        self.mu_steps: List[float] = mu
        self.mu_init: nn.Parameter = nn.Parameter(torch.Tensor([mu[0]]).to(device))
        self.matrix: torch.Tensor = torch.from_numpy(A).float().to(device)
        self.device = device
        self.net: nn.ModuleList = nn.ModuleList([])

        for i in range(numProjections):

            block = Block(A, depth, device)  
            self.net.append(block)

            if i != numProjections - 1:

                # Create a Convolution Layer to mirror the 'Descent' Step
                convolution = nn.Conv2d(in_channels=i + 3, out_channels=1, kernel_size=1)

                # Initialize weights and bias for the convolution layer
                new_weights = init_weights(mu[i + 1], i + 3).type_as(convolution.weight)
                new_bias = torch.zeros_like(convolution.bias)

                convolution.weight.data = new_weights
                convolution.bias.data = new_bias

                self.net.append(convolution)

    def forward(self, y: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the network.

        Parameters
        ----------
        y : torch.Tensor
            The input tensor, typically the observed data.

        Returns
        -------
        List[torch.Tensor]
            The list of intermediate projections through the network layers.
        """
        d: int = y.size(0)
        m: int = int(np.sqrt(self.matrix.size(1)))

        # Initial projection step
        x_t: torch.Tensor = self.mu_init * torch.reshape((self.matrix.t() @ y.t()).t(), (d, 1, m, m))
        history: List[torch.Tensor] = []

        memory: Optional[torch.Tensor] = None

        for i in range(self.numProjections * 2 - 1):
            if i % 2 == 0:
                # Process through Block layers and update memory
                x_t, memory = self.net[i](y, x_t, memory)
                history.append(x_t.reshape(d, int(m ** 2)))
            else:
                # Process through convolutional layers
                x_t = self.net[i](memory)

        return history

    def predict(self, features: np.ndarray, calculate_intermediate: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Make predictions with the network, optionally including intermediate results.

        Parameters
        ----------
        features : np.ndarray
            Input features as a numpy array of shape (input_dim,).
        calculate_intermediate : bool, optional
            If True, returns intermediate projections for each layer.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            The final output as a numpy array or a list of intermediate outputs if calculate_intermediate is True.
        """
        self.eval()   

        features_tensor: torch.Tensor = torch.reshape(
            torch.from_numpy(features).float(), 
            (1, features.shape[0])
        ).to(self.device)

        with torch.no_grad():   
            tensor_hist: List[torch.Tensor] = self.forward(features_tensor)

        if calculate_intermediate:
            # Optionally, return all intermediate image reconstructions
            images_hist: List[np.ndarray] = [tensor.detach().cpu().numpy() for tensor in tensor_hist]
            return images_hist

        # Return the final image Reconstruction
        return tensor_hist[-1].detach().cpu().numpy()