import torch
import torch.nn as nn

def train(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_function: callable, device: torch.device, display: bool = False):
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
    loss_function : callable
        A function that computes the loss given predictions and ground truth images.
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