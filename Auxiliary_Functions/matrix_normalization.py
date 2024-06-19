import numpy as np

def normalize_max_row_norm(matrix: np.ndarray) -> np.ndarray:
    """
    Normalizes the rows of the input matrix by the maximum norm.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix to be normalized. It should be a 2D array.

    Returns
    -------
    np.ndarray
        The matrix with rows normalized by the maximum norm of the rows.
    """
    row_norms = np.linalg.norm(matrix, axis=1)  
    max_norm = np.max(row_norms)                 
    normalized_matrix = matrix / max_norm        

    return normalized_matrix
