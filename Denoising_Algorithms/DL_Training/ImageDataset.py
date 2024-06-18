import numpy as np

from torch.utils.data import Dataset
from typing import Tuple

class ImageDataset(Dataset):
    """
    Pytorch Dataset for Loading Images.

    Attributes
    ----------
    X : np.ndarray
        The observation matrix 
    Y : np.ndarray
        The ground truth matrix 
    """

    def __init__(self, data: dict):
        """
        Initializes the dataset with feature and ground truth data.

        Parameters
        ----------
        data : dict
            A dictionary with keys 'X' and 'Y' where 'X' contains the features and 'Y' contains the ground truths.
        """
        self.X: np.ndarray = data['X'].astype(np.float32)
        self.Y: np.ndarray = data['Y'].astype(np.float32)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the feature and ground truth pair for the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the feature and the ground truth corresponding to the given index.
        """
        return self.X[idx], self.Y[idx]