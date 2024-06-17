import numpy as np

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