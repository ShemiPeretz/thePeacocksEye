import torch
import torch.nn as nn


class TempLSTM(nn.Module):
    """
    LSTM model for temperature forecasting.

    Parameters:
    - input_size: Number of input features.
    - hidden_size: Number of features in the hidden state.
    - output_size: Number of output features.
    - num_layers: Number of recurrent layers (default is 1).
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TempLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        - x: Input tensor of shape (batch_size, input_size).

        Returns:
        - out: Output tensor of shape (batch_size, output_size).
        """
        x = x.unsqueeze(1)  # Add a dimension to match LSTM input requirements

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # Pass the last time step's output through the fully connected layers
        return out


class HumidityLSTM(nn.Module):
    """
    LSTM model for humidity forecasting.

    Parameters:
    - input_size: Number of input features.
    - hidden_size: Number of features in the hidden state.
    - output_size: Number of output features.
    - num_layers: Number of recurrent layers (default is 2).
    - dropout: Dropout probability for regularization (default is 0.2).
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(HumidityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        - x: Input tensor of shape (batch_size, input_size).

        Returns:
        - out: Output tensor of shape (batch_size, output_size).
        """
        x = x.unsqueeze(1)  # Add a dimension to match LSTM input requirements

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # Pass the last time step's output through the fully connected layers
        return out
