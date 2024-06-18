import torch
import torch.nn as nn
from typing import List

def intermediate_scaled_loss(predictions: List[torch.Tensor], true_img: torch.Tensor, omega: float = 1.0) -> torch.Tensor:
    """
    Computes the loss for each intermediate prediction with a decaying factor.

    Parameters
    ----------
    predictions : List[torch.Tensor]
        List of tensors containing the intermediate image predictions.
    true_img : torch.Tensor
        The ground truth image.
    omega : float, optional
        Decay factor for scaling the intermediate losses, default: 1.0.

    Returns
    -------
    torch.Tensor
        The computed loss.
    """
    criterion = nn.MSELoss()
    L = len(predictions)
    loss = 0.0

    for i in range(L):
        decay = omega ** (L - i - 1)
        loss += decay * criterion(predictions[i], true_img)

    return loss

def last_layer_loss(predictions: List[torch.Tensor], true_img: torch.Tensor) -> torch.Tensor:
    """
    Computes the loss using only the final prediction.

    Parameters
    ----------
    predictions : List[torch.Tensor]
        List of tensors containing the intermediate image predictions.
    true_img : torch.Tensor
        The ground truth image.

    Returns
    -------
    torch.Tensor
        The computed loss.
    """
    criterion = nn.MSELoss()

    final_prediction = predictions[-1]
    loss = criterion(final_prediction, true_img)

    return loss

def skipped_layer_loss(predictions: List[torch.Tensor], true_img: torch.Tensor, skip: int = 5) -> torch.Tensor:
    """
    Computes the loss for predictions at every nth layer.

    Parameters
    ----------
    predictions : List[torch.Tensor]
        List of tensors containing the intermediate predictions.
    true_img : torch.Tensor
        The ground truth image.
    skip : int, optional
        Number of layers to skip between each evaluated loss, default: 5.

    Returns
    -------
    torch.Tensor
        The computed loss.
    """
    criterion = nn.MSELoss()
    loss = 0.0

    for i in range(skip - 1, len(predictions), skip):
        loss += criterion(predictions[i], true_img)

    return loss
