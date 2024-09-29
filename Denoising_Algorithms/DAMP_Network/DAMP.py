import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from typing import List, Tuple, Union

class Block(nn.Module):
    """
    A Block within the Unrolled D-AMP model.
    """

    def __init__(self, A: torch.Tensor, mu: float, eps: float, model: nn.Module):
        """
        Initializes the Block with the given parameters.

        Parameters
        ----------
        A : torch.Tensor
            The sensing matrix.
        mu : float
            The step size for the gradient step.
        eps : float
            The small perturbation used in the divergence estimation.
        model : nn.Module
            The deep learning denoiser model.
        """
        super().__init__()

        self.model: nn.Module = model
        self.matrix: torch.Tensor = A
        self.mu_step: nn.Parameter = nn.Parameter(torch.Tensor([mu]))  # Learnable step size  
        self.eps: float = eps

    def forward(self, y: torch.Tensor, x: torch.Tensor, z: torch.Tensor, residual_connection: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the block.

        Parameters
        ----------
        y : torch.Tensor
            The observed data.
        x : torch.Tensor
            The current estimate of x.
        z : torch.Tensor
            The current residual.
        residual_connection : bool
            Whether the projection step predicts the image or the residual

        Returns
        -------
        x_new : torch.Tensor
            The updated estimate of x.
        z_new : torch.Tensor
            The updated residual.
        """
        d: int = y.size(0)   
        m: int = self.matrix.size(0) 
        n: int = self.matrix.size(1)
        n_sqrt: int = int(np.sqrt(n))  

        x_temp: torch.Tensor = x + self.mu_step * (self.matrix.t() @ z.t()).t()  
        x_rshp: torch.Tensor = x_temp.view(d, 1, n_sqrt, n_sqrt)

        if residual_connection:
            x_new = torch.reshape(x_rshp, (d, n_sqrt * n_sqrt)) + self.model(x_rshp)
        else:
            x_new = self.model(x_rshp)

        # Estimate Divergence via Monte Carlo
        b = torch.randn_like(x_new)   

        x_temp_est = x_temp + self.eps * b
        x_temp_est = x_temp_est.view(d, 1, n_sqrt, n_sqrt)

        if residual_connection:
            x_div = torch.reshape(x_temp_est, (d, n_sqrt * n_sqrt)) + self.model(x_temp_est)
        else:
            x_div = self.model(x_temp_est)

        div = torch.sum(b * (x_div - x_new), dim=1) / self.eps  # Shape: (d,)
        OCT = z * (div.view(-1, 1) / m)  # Shape: (d, m)
        z_new = y - (self.matrix @ x_new.t()).t() + OCT

        return x_new, z_new

class DAMP(nn.Module):
    """
    Unrolled D-AMP network model.
    """

    def __init__(self, A: np.ndarray, mu: List[float], eps: float, numProjections: int, model: nn.Module, device: torch.device, residual: bool = False):
        """
        Initializes the DAMP model with the given parameters.

        Parameters
        ----------
        A : np.ndarray
            The sensing matrix.
        mu : List[float]
            A list of step sizes for each projection.
        eps : float
            The small perturbation used in the divergence estimation.
        numProjections : int
            Number of projections in the network.
        model : nn.Module
            The deep learning denoiser model.
        device : torch.device
            The device on which to perform computations (CPU or GPU).
        residual : bool
            Whether the projection step predicts the image or the residual, default : False
        """
        super().__init__()

        self.matrix: torch.Tensor = torch.from_numpy(A).float().to(device)
        self.numProjections: int = numProjections
        self.eps: float = eps
        self.mu: List[float] = mu
        self.device = device
        self.residual = residual

        if len(mu) != numProjections:
            raise ValueError("Length of mu must be equal to numProjections")

        self.net: nn.ModuleList = nn.ModuleList([
            Block(self.matrix, mu[i], self.eps, deepcopy(model)) for i in range(numProjections)
        ])

    def forward(self, y: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the network.

        Parameters
        ----------
        y : torch.Tensor
            The observed data.

        Returns
        -------
        history : List[torch.Tensor]
            A list of intermediate tensors from each projection.
        """

        d: int = y.size(0)  # batch size
        n: int = self.matrix.size(1)  # number of columns in the sensing matrix

        x_t: torch.Tensor = torch.zeros(d, n).to(self.device)
        z_t: torch.Tensor = y.clone()

        history: List[torch.Tensor] = []

        # Process input iteratively through each projection/block
        for block in self.net:
            x_t, z_t = block(y, x_t, z_t, self.residual)
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

        with torch.no_grad():
            tensor_hist: List[torch.Tensor] = self.forward(features_tensor)

        if calculate_intermediate:
            # Optionally, return all intermediate image reconstructions
            images_hist: List[np.ndarray] = [tensor.detach().cpu().numpy() for tensor in tensor_hist]

            return images_hist

        return tensor_hist[-1].detach().cpu().numpy()
