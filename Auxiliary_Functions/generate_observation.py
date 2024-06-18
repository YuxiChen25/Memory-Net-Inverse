import numpy as np
from typing import Dict, Tuple

def generate_observations(dataset: np.ndarray, A: np.ndarray, sigma: float = None) -> dict:
    """
    Generates compressed observations using a sensing matrix A and adds Gaussian noise if sigma is provided.

    Parameters
    ----------
    dataset : np.ndarray
        The original image dataset where each row is an image vector
    A : np.ndarray
        The sensing matrix used for compression.
    sigma : float, optional
        The standard deviation of the Gaussian noise to be added to the compressed observations.
        If not provided, no noise will be added.

    Returns
    -------
    dict
        A dictionary containing the compressed observations "X" and the original dataset "Y".
    """

    n, _ = dataset.shape
    m, _ = A.shape

    X = np.zeros((n, m))
    Y = dataset / 255   

    for i, x in enumerate(Y):

        y = A @ x
        
        if sigma is not None:

            rng = np.random.default_rng(i)  # Create a new RNG with a seed based on the image index
            noise = sigma * rng.normal(0, 1, size=(m,))
            X[i] = y + noise

        else:

            X[i] = y

    return {"X": X, "Y": Y}


def split_dataset(data: Dict[str, np.ndarray], train_ratio: float = 0.8, seed: int = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Splits the dataset into training and test sets with an optional seed for reproducibility.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        A dictionary containing 'X' for observed data and 'Y' for ground truth data.
    train_ratio : float, optional
        The proportion of the data to be used for training. Default is 0.8 (80%).
    seed : int, optional
        The seed for the random number generator to ensure reproducibility. Default is None.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        A tuple containing two dictionaries: the first for the training set and the second for the test set.
        Each dictionary has keys 'X' and 'Y'.
    """
    X = data['X']
    Y = data['Y']
    n_samples = X.shape[0]

    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Randomly Permute the Entries
    idxs = rng.permutation(n_samples)  
    
    n_train = int(train_ratio * n_samples)  
    train_idxs = idxs[:n_train]
    test_idxs  = idxs[n_train:]

    # Train-Test Split
    X_train = X[train_idxs]
    Y_train = Y[train_idxs]
    X_test  = X[test_idxs]
    Y_test  = Y[test_idxs]

    train_set = {"X": X_train, "Y": Y_train}
    test_set  = {"X": X_test, "Y": Y_test}

    return train_set, test_set
