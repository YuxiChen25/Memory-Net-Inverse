import os
import shutil
import pickle

from typing import Any

def save_progress(base_path: str, weights_path: str, history: Any, subdir: str):
    """
    Saves the best model weights and training history to the specified directory.

    Parameters
    ----------
    proj_path : str
        The base path
    weights_path : str
        The path to the model weights file that needs to be saved.
    history : Any
        The training history object to be saved.
    subdir : str, optional
        The sub-directory name within the project path to save the files.
    """

    target_dir = os.path.join(base_path, subdir)
    
    os.makedirs(target_dir, exist_ok=True)
    
    shutil.copy(weights_path, target_dir)
    
    history_path = os.path.join(target_dir, 'history.pkl')

    with open(history_path, 'wb') as f:
        pickle.dump(history, f)