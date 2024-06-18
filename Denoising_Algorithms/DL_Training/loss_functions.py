import torch
import torch.nn as nn

from typing import List
from abc import ABC, abstractmethod

class BaseLoss(ABC):
    """
    Abstract base class for loss functions.
    """

    @abstractmethod
    def __call__(self, predictions: List[torch.Tensor], true_img: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        predictions : List[torch.Tensor]
            List of tensors containing the intermediate predictions.
        true_img : torch.Tensor
            The ground truth image.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        pass

class IntermediateScaledLoss(BaseLoss):
    """
    Computes the loss for each intermediate prediction with a decaying factor.
    """

    def __init__(self, omega: float = 1.0):
        """
        Initialize the intermediate scaled loss with a decay factor.

        Parameters
        ----------
        omega : float, optional
            Decay factor for scaling the intermediate losses, by default 1.0.
        """
        super().__init__()
        self.criterion = nn.MSELoss()
        self.omega = omega

    def __call__(self, predictions: List[torch.Tensor], true_img: torch.Tensor) -> torch.Tensor:
        """
        Compute the scaled loss for all intermediate predictions.

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
        L = len(predictions)
        loss = 0.0
        for i in range(L):
            decay = self.omega ** (L - i - 1)
            loss += decay * self.criterion(predictions[i], true_img)
        return loss

class LastLayerLoss(BaseLoss):
    """
    Computes the loss using only the final prediction.
    """

    def __init__(self):
        """
        Initialize the last layer loss.
        """
        super().__init__()
        self.criterion = nn.MSELoss()

    def __call__(self, predictions: List[torch.Tensor], true_img: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss using only the final prediction.

        Parameters
        ----------
        predictions : List[torch.Tensor]
            List of tensors containing the intermediate predictions.
        true_img : torch.Tensor
            The ground truth image.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        final_prediction = predictions[-1]
        loss = self.criterion(final_prediction, true_img)
        return loss

class SkippedLayerLoss(BaseLoss):
    """
    Computes the loss for predictions at every nth layer.
    """

    def __init__(self, skip: int = 5):
        """
        Initialize the skipped layer loss.

        Parameters
        ----------
        skip : int, optional
            Number of layers to skip between each evaluated loss, by default 5.
        """
        super().__init__()
        self.criterion = nn.MSELoss()
        self.skip = skip

    def __call__(self, predictions: List[torch.Tensor], true_img: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for predictions at every nth layer.

        Parameters
        ----------
        predictions : List[torch.Tensor]
            List of tensors containing the intermediate predictions.
        true_img : torch.Tensor
            The ground truth image.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        loss = 0.0
        for i in range(self.skip - 1, len(predictions), self.skip):
            loss += self.criterion(predictions[i], true_img)
        return loss
