import numpy as np
from typing import Tuple, List

def gaussian_matrix(shape: Tuple[int, int], seed: int = None) -> np.ndarray:
    """
    Constructs a Gaussian sensing matrix A where each element A_ij is drawn from a Gaussian distribution N(0, 1/m).

    Parameters
    ----------
    shape : Tuple[int, int]
        The shape of the matrix A as (rows, columns).
    seed : int, optional
        The seed value for the internal NumPy random number generator. If not provided, the RNG will be initialized
        with a random seed value.

    Returns
    -------
    np.ndarray
        The constructed Gaussian sensing matrix A.
    """

    if shape[0] > shape[1]:
        raise ValueError("Row count greater than column count")
    
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1 / np.sqrt(shape[0]), shape)

def dct_matrix(n: int) -> np.ndarray:
    """
    Generates a Discrete Cosine Transform (DCT) matrix of size n x n.

    Parameters
    ----------
    n : int
        The size of the DCT matrix.

    Returns
    -------
    np.ndarray
        The DCT matrix of size n x n.
    """
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    D = np.sqrt(2 / n) * np.cos(np.pi * (2 * x + 1) * y / (2 * n))
    D[0, :] *= np.sqrt(1 / 2)
    return D

def kron_product_1d_dct(n: int) -> np.ndarray:
    """
    Computes the Kronecker product of two 1D DCT matrices of size n x n.

    Parameters
    ----------
    n : int
        The size of the DCT matrix.

    Returns
    -------
    np.ndarray
        The Kronecker product of two 1D DCT matrices of size n^2 x n^2.
    """
    D = dct_matrix(n)
    return np.kron(D, D)

def generate_103_idxs(n: int) -> List[int]:
    """
    Generates a list of 103 indices located on the top left corner
    of an image for the subsampling matrix in the DCT setting. 

    Parameters
    ----------
    n : int
        The length of the image (assumed to be square).

    Returns
    -------
    List[int]
        A list of indices 
    """

    indices = []

    for col in range(10):
        for row in range(10):
            indices.append(row * n + col)

    for col in range(3):
        indices.append(10 * n + col)

    return indices

def generate_250_idxs(n: int) -> List[int]:
    """
    Generates a list of 250 indices located on the top left corner
    of an image for the subsampling matrix in the DCT setting. 

    Parameters
    ----------
    n : int
        The length of the image (assumed to be square).

    Returns
    -------
    List[int]
        A list of indices
    """

    indices = []

    for col in range(16):

        if col < 15:
            for row in range(16):
                indices.append(row * n + col)
        else:
            for row in range(10):
                indices.append(row * n + col)

    return indices

def generate_410_idxs(n: int) -> List[int]:
    """
    Generates a list of 410 indices located on the top left corner
    of an image for the subsampling matrix in the DCT setting. 

    Parameters
    ----------
    n : int
        The length of the image (assumed to be square).

    Returns
    -------
    List[int]
        A list of indices 
    """

    indices = []

    for col in range(20):
        for row in range(20):
            indices.append(row * n + col)

    for col in range(10):
        indices.append(20 * n + col)

    return indices

def create_sampling_matrix_32(n: int, k: int, seed: int = None) -> np.ndarray:
    """
    Creates a DCT sampling matrix based on fixed indices and
    randomly sampled additional indices for 32 x 32 images

    Parameters
    ----------
    n : int
        The length of the image (assumed to be square).
    k : int
        The total number of indices to include in the sampling matrix.
    seed : int, optional
        The seed value for the random number generator. If not provided, a random seed will be used.

    Returns
    -------
    np.ndarray
        The sampling matrix of size total_k x (n^2).
    """
    
    total_pixels = n * n

    if k > total_pixels:
        raise ValueError("total_k must be less than or equal to total pixels in the image")
    
    S = np.zeros((k, total_pixels))
    fixed_idxs = generate_103_idxs(n)
    other_idxs = set(range(total_pixels)) - set(fixed_idxs)

    rng = np.random.default_rng(seed)
    
    if k > 103:
        extra_idxs = rng.choice(list(other_idxs), k - 103, replace=False).tolist()
    else:
        extra_idxs = []
    
    all_idxs = sorted(fixed_idxs + extra_idxs)
    
    for i, index in enumerate(all_idxs):
        S[i, index] = 1

    return S

def create_sampling_matrix_50(n: int, k: int, seed: int = None) -> np.ndarray:
    """
    Creates a DCT sampling matrix based on fixed indices and
    randomly sampled additional indices for 50 x 50 images

    Parameters
    ----------
    n : int
        The length of the image (assumed to be square).
    k : int
        The total number of indices to include in the sampling matrix.
    seed : int, optional
        The seed value for the random number generator. If not provided, a random seed will be used.

    Returns
    -------
    np.ndarray
        The sampling matrix of size total_k x (n^2).
    """
    
    total_pixels = n * n

    if k > total_pixels:
        raise ValueError("total_k must be less than or equal to total pixels in the image")
    
    S = np.zeros((k, total_pixels))
    fixed_idxs = generate_250_idxs(n)
    other_idxs = set(range(total_pixels)) - set(fixed_idxs)

    rng = np.random.default_rng(seed)
    
    if k > 250:
        extra_idxs = rng.choice(list(other_idxs), k - 250, replace=False).tolist()
    else:
        extra_idxs = []
    
    all_idxs = sorted(fixed_idxs + extra_idxs)
    
    for i, index in enumerate(all_idxs):
        S[i, index] = 1

    return S

def create_sampling_matrix_64(n: int, k: int, seed: int = None) -> np.ndarray:
    """
    Creates a DCT sampling matrix based on fixed indices and
    randomly sampled additional indices for 64 x 64 images

    Parameters
    ----------
    n : int
        The length of the image (assumed to be square).
    k : int
        The total number of indices to include in the sampling matrix.
    seed : int, optional
        The seed value for the random number generator. If not provided, a random seed will be used.

    Returns
    -------
    np.ndarray
        The sampling matrix of size total_k x (n^2).
    """
    
    total_pixels = n * n

    if k > total_pixels:
        raise ValueError("total_k must be less than or equal to total pixels in the image")
    
    S = np.zeros((k, total_pixels))
    fixed_idxs = generate_410_idxs(n)
    other_idxs = set(range(total_pixels)) - set(fixed_idxs)

    rng = np.random.default_rng(seed)
    
    if k > 410:
        extra_idxs = rng.choice(list(other_idxs), k - 410, replace=False).tolist()
    else:
        extra_idxs = []
    
    all_idxs = sorted(fixed_idxs + extra_idxs)
    
    for i, index in enumerate(all_idxs):
        S[i, index] = 1

    return S