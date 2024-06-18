import torch
import numpy as np
import torch.nn as nn

from typing import List, Dict
from skimage.metrics import peak_signal_noise_ratio as psnr

def average_PSNR(model: nn.Module, directory: str, numProjections: int, test_set: Dict[str, np.ndarray], print_psnr: bool = False) -> List[float]:
    """
    Calculate the average PSNR for each projection layer of the model across a test set.

    Parameters
    ----------
    model : nn.Module
        The neural network model to evaluate.
    directory : str
        The path to the saved model state dictionary.
    numProjections : int
        The number of projection layers to evaluate in the model.
    test_set : Dict[str, np.ndarray]
        A dictionary containing 'X' for observed data and 'Y' for ground truth data.
    print_psnr : bool, optional
        If True, prints the PSNR for each layer and each image. Default is False.

    Returns
    -------
    List[float]
        A list containing the average PSNR for each projection layer across the test set.
    """

    model.load_state_dict(torch.load(directory, map_location=model.device))
    model.eval()  

    X = test_set['X']  # Compressed images 
    Y = test_set['Y']  # Ground truth images
    n_samples = X.shape[0]

    cumulative_PSNR = [0.0] * numProjections

    with torch.no_grad():   

        for i in range(n_samples):

            compressed = X[i]
            true_image = Y[i]

            predictions = model.predict(compressed, calculate_intermediate=True)

            for j, prediction in enumerate(predictions):

                pred_flat = prediction.flatten() # reshape to (n, )
                true_flat = true_image.flatten()  

                psnr_value = psnr(true_flat, pred_flat, data_range=1.0)
                cumulative_PSNR[j] += psnr_value

    # Compute average PSNR for each projection layer
    average_PSNRs = [cumulative / n_samples for cumulative in cumulative_PSNR]

    if print_psnr:
        for layer_idx, avg_PSNR in enumerate(average_PSNRs):
            print(f"Average PSNR for Projection Layer {layer_idx} across {n_samples} images: {avg_PSNR:.2f}")

    return average_PSNRs
