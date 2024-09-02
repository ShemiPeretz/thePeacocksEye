import torch
import torch.nn as nn
from model import TempLSTM
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pytest

def rolling_forecast_test(model, test_loader, criterion, num_steps=3):
    """
    Performs a rolling forecast test on the provided model.

    Parameters:
    - model: The trained LSTM model for forecasting.
    - test_loader: DataLoader containing the test dataset.
    - criterion: Loss function used to compute the error (e.g., MSELoss).
    - num_steps: Number of steps ahead for which to forecast.

    Returns:
    - avg_mse_per_step: List containing the average MSE for each forecasting step.
    """
    model.eval()  # Set the model to evaluation mode
    mse_per_step = [0.0] * num_steps
    count_per_step = [0] * num_steps

    with torch.no_grad():  # Disable gradient computation for testing
        for X_batch, y_batch in test_loader:
            for i in range(X_batch.size(0)):
                input_seq = X_batch[i]
                true_values = y_batch[i:i + num_steps]
                predictions = []

                # Perform rolling forecast
                for step in range(num_steps):
                    predicted_output = model(input_seq.unsqueeze(0))
                    predictions.append(predicted_output)
                    input_seq = torch.cat([input_seq[1:], predicted_output.squeeze(0)], dim=0)

                # Calculate MSE for each step
                for step in range(num_steps):
                    if step < len(true_values):
                        mse = criterion(predictions[step], true_values[step].unsqueeze(0))
                        mse_per_step[step] += mse.item()
                        count_per_step[step] += 1

    avg_mse_per_step = [mse_per_step[i] / count_per_step[i] if count_per_step[i] > 0 else 0.0 for i in range(num_steps)]
    return avg_mse_per_step

def test_model(model, test_loader, criterion):
    """
    Evaluates the model on the test dataset and plots the loss per batch.

    Parameters:
    - model: The trained LSTM model for forecasting.
    - test_loader: DataLoader containing the test dataset.
    - criterion: Loss function used to compute the error (e.g., MSELoss).

    Returns:
    - test_loss: The average loss over the entire test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    batch_num = 0
    max_loss = 0
    losses = []

    with torch.no_grad():  # Disable gradient computation for testing
        for X_batch, y_batch in test_loader:
            batch_num += 1
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            if torch.isnan(loss):
                continue  # Skip NaN losses
            test_loss += loss.item() * X_batch.size(0)
            max_loss = max(max_loss, loss.item())
            losses.append(loss.item())

    print(f'Max Loss: {max_loss}')
    test_loss /= len(test_loader.dataset)

    # Plot the loss for each batch
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Test Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

    return test_loss

# PyTest fixtures to set up model, test data, and criterion
@pytest.fixture
def model():
    """
    Fixture for loading the pre-trained LSTM model for humidity forecasting.
    """
    input_size = 421
    hidden_size = 421
    output_size = 1
    num_layers = 2
    model = TempLSTM(input_size, hidden_size, output_size, num_layers)
    model.load_state_dict(torch.load('Humidity_forecast_model_combined_loss.pt'))
    return model

@pytest.fixture
def test_loader():
    """
    Fixture for loading the test dataset for humidity forecasting.
    """
    batch_size = 64
    X_test = torch.load('X_test_Relative humidity (%).pt')
    y_test = torch.load('y_test_Relative humidity (%).pt')
    test_dataset = TensorDataset(X_test, y_test)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

@pytest.fixture
def criterion():
    """
    Fixture for the loss function used in testing (Mean Squared Error Loss).
    """
    return nn.MSELoss()

# PyTest function to test the model using the above fixtures
def test_model_pytest(model, test_loader, criterion):
    """
    PyTest function to evaluate the model and assert that the loss is positive.

    Parameters:
    - model: The trained LSTM model for forecasting.
    - test_loader: DataLoader containing the test dataset.
    - criterion: Loss function used to compute the error (e.g., MSELoss).
    """
    test_loss = test_model(model, test_loader, criterion)
    assert test_loss > 0, "Test loss should be positive"
    print(f'Test Loss: {test_loss:.4f}')

if __name__ == "__main__":
    # This block will not be executed when running with pytest
    input_size = 421
    hidden_size = 421
    output_size = 1
    num_layers = 2
    model = TempLSTM(input_size, hidden_size, output_size, num_layers)
    model.load_state_dict(torch.load('Humidity_forecast_model_combined_loss.pt'))
    model.eval()

    X_test = torch.load('X_test_Relative humidity (%).pt')
    y_test = torch.load('y_test_Relative humidity (%).pt')
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    criterion = nn.MSELoss()

    test_loss = test_model(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}')

    num_steps = 3
    mse_per_step = rolling_forecast_test(model, test_loader, criterion, num_steps)
    for step in range(num_steps):
        print(f'MSE for step {step + 1}: {mse_per_step[step]:.4f}')