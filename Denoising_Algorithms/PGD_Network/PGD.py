import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from typing import List, Tuple, Union

class Block(nn.Module):
    """
    A Block within the EndToEnd model consisting of convolutional layers and a gradient step.

    Attributes
    ----------
    model : torch.nn.Sequential
        Sequential container for the convolutional layers.
    mu_step : torch.nn.Parameter
        Parameter for the step size in the gradient step.
    matrix : torch.Tensor
        The sensing matrix used in the block.
    """

    def __init__(self, A: torch.tensor, mu: float, model: nn.Module):
        """
        Initializes the Block with the given parameters.

        Parameters
        ----------
        A : np.ndarray
            The sensing matrix.
        mu : float
            The step size for the gradient step.
        model : nn.Module
            The deep learning projector model
        """
        super().__init__()

        self.model: nn.Module = model
        self.mu_step: nn.Parameter = nn.Parameter(torch.Tensor([mu]))  # Learnable step size  
        self.matrix: torch.Tensor = A

    def forward(self, y: torch.Tensor, x: torch.Tensor, residual: bool) -> torch.Tensor:
        """
        Forward pass through the block.

        Parameters
        ----------
        y : torch.Tensor
            The input tensor, typically the observed data.
        x : torch.Tensor
            The initial tensor to be updated in this block.
        residual : bool
            Whether the projection step predicts the image or the residual

        Returns
        -------
        torch.Tensor
            The output tensor after processing through the block.
        """
        d: int = y.size(0)
        m: int = int(np.sqrt(x.size(1)))

        # Gradient Step
        residual: torch.Tensor = y - (self.matrix @ x.t()).t()
        x_tilde: torch.Tensor = x + self.mu_step * (self.matrix.t() @ residual.t()).t()
        x_tilde = torch.reshape(x_tilde, (d, 1, m, m))  # Reshape x_tilde for the convolutional layers

        if residual: 
            return torch.reshape(x_tilde, (d, m * m)) + self.model(x_tilde)
        
        return self.model(x_tilde)
    
class PGD(nn.Module):
    """
    End-to-End PGD network model that processes input through multiple layers of convolutional operations,
    specifically tailored for projected gradient descent.

    Attributes
    ----------
    matrix : torch.Tensor
        The sensing matrix used in the model.
    numProjections : int
        Number of projections (layers) in the network.
    mu : List[float]
        List of step sizes for gradient descent steps in each projection.
    device : torch.device
        The device on which to perform computations (CPU or GPU).
    residual : bool
        Whether the projection step predicts the image or the residual 
    net : torch.nn.ModuleList
        List containing all the Block layers.
    """

    def __init__(self, A: np.ndarray, mu: List[float], numProjections: int, model: nn.Module, device: torch.device, residual: bool = False):
        """
        Initializes the EndToEnd model with the given parameters.

        Parameters
        ----------
        A : np.ndarray
            The sensing matrix.
        mu : List[float]
            A list of step sizes for each projection.
        numProjections : int
            Number of projections in the network.
        model : nn.Module
            The deep learning projector model.
        device : torch.device
            The device on which to perform computations (CPU or GPU).
        residual : bool
            Whether the projection step predicts the image or the residual, default : False
        """
        super().__init__()

        self.matrix: torch.Tensor = torch.from_numpy(A).float().to(device)
        self.numProjections: int = numProjections
        self.mu: List[float] = mu
        self.device = device
        self.residual = residual

        if len(mu) != numProjections:
            raise ValueError("Length of mu must be equal to numProjections")

        self.net: nn.ModuleList = nn.ModuleList([
            Block(self.matrix, mu[i], deepcopy(model)) for i in range(numProjections)
        ])

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the network.

        Parameters
        ----------
        y : torch.Tensor
            The input tensor, typically the observed data.

        Returns
        -------
        history : List[torch.Tensor]
            A list of intermediate tensors from each projection.
        """

        d: int = y.size(0) # batch size
        m_sq: int = self.matrix.size(1)  # number of columns in the sensing matrix

        x_t: torch.Tensor = torch.zeros(d, m_sq).to(self.device)
        history: List[torch.Tensor] = []   

        # Process input iteratively through each projection/block
        for block in self.net:
            x_t = block(y, x_t, self.residual)
            history.append(x_t)  

        return history

    def predict(self, features: np.ndarray, calculate_intermediate: bool) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Make predictions with the network, optionally including intermediate results.

        Parameters
        ----------
        features : np.ndarray
            Input features as a numpy array of shape (input_dim,).
        calculate_intermediate : bool
            Boolean indicating whether to return intermediate results.

        Returns
        -------
        Union[np.ndarray, List[np.ndarray]]
            If calculate_intermediate is True, returns a list of intermediate outputs as numpy arrays.
            Otherwise, returns the final output as a numpy array.
        """
        self.eval()   

        features_tensor: torch.Tensor = torch.reshape(
            torch.from_numpy(features).float(), 
            (1, features.shape[0])
        ).to(self.device)

        tensor_hist: List[torch.Tensor] = self.forward(features_tensor)

        if calculate_intermediate:
            # Optionally, return all intermediate image reconstructions
            images_hist: List[np.ndarray] = [tensor.detach().cpu().numpy() for tensor in tensor_hist]

            return images_hist

        return tensor_hist[-1].detach().cpu().numpy()
    
