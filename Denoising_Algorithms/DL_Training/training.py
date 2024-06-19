import os
import time
import torch
import numpy as np
import torch.nn as nn

from typing import List, Tuple
from torch.utils.data import random_split, DataLoader

from Denoising_Algorithms.PGD_Network.PGD import PGD
from Denoising_Algorithms.DL_Training.ImageDataset import ImageDataset
from Denoising_Algorithms.DL_Training.loss_functions import BaseLoss, LastLayerLoss
from Denoising_Algorithms.Memory_Network.Memory_Net import MemoryNetwork

def train(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_function: BaseLoss, device: torch.device, display: bool = False) -> float:
    """
    Trains the model using the provided loss function.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    loader : torch.utils.data.DataLoader
        DataLoader for the training data.
    optimizer : torch.optim.Optimizer
        Optimizer for the model parameters.
    loss_function : BaseLoss
        An instance of a loss class that computes the loss given predictions and ground truth images.
    device : torch.device
        The device to perform computations on.
    display : bool
        Whether to display the current test loss, default: False. 

    Returns
    -------
    float
        The average loss per sample over the training data.
    """
    model.train()

    train_loss = 0.0
    train_data = 0

    for features, true_img in loader:

        features, true_img = features.to(device), true_img.to(device)

        history = model.forward(features)

        loss = loss_function(history, true_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * features.size(0)
        train_data += features.size(0)

    average_loss = train_loss / train_data

    if display:

        print(f'Average training loss: {average_loss}')

    return average_loss

def test(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, display: bool = False) -> float:
    """
    Evaluates the model on the provided data loader, computing the final output loss.

    Parameters
    ----------
    model : nn.Module
        The neural network model to evaluate.
    loader : torch.utils.data.DataLoader
        DataLoader for the testing or validation data.
    device : torch.device
        The device to perform computations on.
    display : bool, optional
        Whether to display the current test/validation loss, by default False.

    Returns
    -------
    float
        The average loss per sample over the test data.
    """
    model.eval()
    criterion = nn.MSELoss()
    
    test_loss = 0.0
    test_data = 0

    with torch.no_grad():

        for features, true_img in loader:

            features, true_img = features.to(device), true_img.to(device)
            final_prediction = model.forward(features)[-1]

            loss = criterion(final_prediction, true_img)
            test_loss += loss.item() * features.size(0)
            test_data += features.size(0)

    average_loss = test_loss / test_data

    if display:

        print(f'Average test loss: {average_loss}')

    return average_loss

def train_main(A: np.ndarray, mu: List[float], 
               train_data: np.ndarray, 
               model: nn.Sequential, 
               device: torch.device, 
               model_type: str = 'PGD', 
               split: float = 0.2, 
               numProjections: int = 10, 
               batch_size: int = 32, 
               epochs: int = 300, 
               residual: bool = False, 
               loss_function: BaseLoss = LastLayerLoss(), 
               model_dir: str = None, 
               display: bool = False) -> Tuple[str, dict]:
    """
    Trains the specified model on the provided dataset.

    Parameters
    ----------
    A : np.ndarray
        The sensing matrix.
    mu : List[float]
        A list of step sizes for each projection step.
    train_data : np.ndarray
        The training data.
    model : nn.Sequential
        The convolutional model to be used within each Block.
    device : torch.device
        The device on which to perform computations (CPU or GPU).
    model_type : str, optional
        The type of model to train ('PGD' for Projected Gradient Descent or 'Memory' for MemoryNetwork), default: 'PGD'.
    split : float, optional
        Fraction of data to be used for validation, default: 0.2.
    numProjections : int, optional
        Number of projection steps in the network, default: 10.
    batch_size : int, optional
        Batch size for training, default: 32.
    epochs : int, optional
        Number of training epochs, default: 300.
    residual : bool, optional
        Whether the projection step predicts the image or the residual, default: False.
    loss_function : BaseLoss, optional
        An instance of a loss class to compute the loss, default: LastLayerLoss.
    model_dir : str, optional
        Directory to save models, default: None (generates a timestamped directory).
    display : bool, optional
        Whether to display real-time training and validation progress, default: False.

    Returns
    -------
    Tuple[str, dict]
        Path to the best model saved based on validation loss and a dictionary containing the training and validation loss history.
    """
    # Split the dataset into training and validation sets
    dataset = ImageDataset(train_data)
    train_size = int((1 - split) * len(dataset))
    valid_size = len(dataset) - train_size

    train_set, valid_set = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

    if display:
        print(f"Validation Size: {valid_size}, Train Size: {train_size}")

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, shuffle=True, batch_size=batch_size)

    if model_dir is None:
        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = os.path.join('models', timestr)
    os.makedirs(model_dir, exist_ok=True)

    best_model_path = None
    min_loss = float('inf')
    train_losses = []
    valid_losses = []

    if model_type == 'PGD':
        chosen_model = PGD(A, mu, numProjections=numProjections, model=model, device=device, residual=residual).to(device)
    elif model_type == 'Memory':
        chosen_model = MemoryNetwork(A, mu, numProjections=numProjections, model=model, device=device, residual=residual).to(device)
    else:
        raise ValueError("Invalid model_type. Choose 'PGD' for Projected Gradient Descent or 'Memory' for MemoryNetwork.")

    # Adam Optimizer
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(chosen_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        if display:
            print(f'Epoch {epoch + 1}/{epochs}')

        # Training step
        train_loss = train(chosen_model, train_loader, optimizer, loss_function, device, display=display)
        train_losses.append(train_loss)

        # Validation step
        valid_loss = test(chosen_model, valid_loader, device, display=display)
        valid_losses.append(valid_loss)

        # Current model directory
        epoch_model_dir = os.path.join(model_dir, f'Epoch_{epoch + 1}_loss_{valid_loss:.4f}')
        os.makedirs(epoch_model_dir, exist_ok=True)

        # Save model weights
        model_save_path = os.path.join(epoch_model_dir, 'model.pth')
        torch.save(chosen_model.state_dict(), model_save_path)

        # Update best model path
        if valid_loss < min_loss:
            min_loss = valid_loss
            best_model_path = model_save_path

    loss_history = {
        'train_losses': train_losses,
        'valid_losses': valid_losses
    }

    return best_model_path, loss_history
