import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import TempLSTM
import torch.nn.functional as F

def cosine_similarity_loss(output, target):
    """
    Calculates the cosine similarity loss between the output and the target.

    Parameters:
    - output: Tensor representing the model's output.
    - target: Tensor representing the target values. If None, uses the output as the target.

    Returns:
    - loss: The cosine similarity loss.
    """
    if target is None:
        target = output
    output = F.normalize(output, p=2, dim=1)
    target = F.normalize(target, p=2, dim=1)
    cosine_similarity = torch.sum(output * target, dim=1)
    loss = 1 - cosine_similarity.mean()
    return loss

def train_model_with_cosine_loss(model, train_loader, val_loader, criterion, optimizer, num_epochs, alpha=0.1):
    """
    Trains the LSTM model using a combination of MSE and cosine similarity losses.

    Parameters:
    - model: The LSTM model to be trained.
    - train_loader: DataLoader for the training dataset.
    - val_loader: DataLoader for the validation dataset.
    - criterion: The loss function (e.g., MSELoss).
    - optimizer: Optimizer for model training (e.g., Adam).
    - num_epochs: Number of epochs to train the model.
    - alpha: Weight for the cosine similarity loss in the combined loss function.

    Returns:
    - train_losses: List of training losses over epochs.
    - val_losses: List of validation losses over epochs.
    """
    train_losses, val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            lstm_out, _ = model.lstm(X_batch)
            outputs = model(X_batch)

            # Calculate MSE loss
            mse_loss = criterion(outputs, y_batch)

            # Calculate cosine similarity loss
            cosine_loss = cosine_similarity_loss(lstm_out, X_batch)

            # Combine losses
            combined_loss = mse_loss + alpha * cosine_loss

            # Backward pass and optimization
            combined_loss.backward()
            optimizer.step()

            train_loss += combined_loss.item() * X_batch.size(0)

        # Average train loss over all batches
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                lstm_out, _ = model.lstm(X_batch)
                outputs = model(X_batch)

                # Calculate validation losses
                mse_loss = criterion(outputs, y_batch)
                cosine_loss = cosine_similarity_loss(lstm_out, X_batch)
                combined_loss = mse_loss + alpha * cosine_loss

                val_loss += combined_loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    return train_losses, val_losses

def train_model_mse_only(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Trains the LSTM model using MSE loss only.

    Parameters:
    - model: The LSTM model to be trained.
    - train_loader: DataLoader for the training dataset.
    - val_loader: DataLoader for the validation dataset.
    - criterion: The loss function (e.g., MSELoss).
    - optimizer: Optimizer for model training (e.g., Adam).
    - num_epochs: Number of epochs to train the model.

    Returns:
    - train_losses: List of training losses over epochs.
    - val_losses: List of validation losses over epochs.
    """
    train_losses, val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)

            # Calculate MSE loss
            mse_loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            mse_loss.backward()
            optimizer.step()

            train_loss += mse_loss.item() * X_batch.size(0)

        # Average train loss over all batches
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)

                # Calculate validation MSE loss
                mse_loss = criterion(outputs, y_batch)
                val_loss += mse_loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    return train_losses, val_losses

if __name__ == "__main__":
    # Load preprocessed data
    X_train = torch.load('X_train_Temperature (°C).pt')
    X_test = torch.load('X_test_Temperature (°C).pt')
    y_train = torch.load('y_train_Temperature (°C).pt')
    y_test = torch.load('y_test_Temperature (°C).pt')

    # Print shapes for debugging
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Split train dataset into train and validation sets
    train_split = int(0.8 * len(X_train))
    X_train_tensor, X_val_tensor = X_train[:train_split], X_train[train_split:]
    y_train_tensor, y_val_tensor = y_train[:train_split], y_train[train_split:]

    # Create datasets from tensors
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test, y_test)

    # Hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.0001  # Reduced learning rate
    alpha = 0.1  # Weight for cosine similarity loss

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model parameters
    input_size = 421
    hidden_size = 421
    output_size = 1
    num_layers = 1

    # Initialize the TempLSTM model
    model = TempLSTM(input_size, hidden_size, output_size, num_layers)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train with combined loss (MSE + Cosine Similarity)
    print("Training with combined loss (MSE + Cosine Similarity):")
    train_losses_combined, val_losses_combined = train_model_with_cosine_loss(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, alpha
    )

    # Save the trained model
    torch.save(model.state_dict(), 'Temperature_forecast_model_combined_loss.pt')

    # Plot losses for combined loss training
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_combined, label='Training Loss (Combined)')
    plt.plot(val_losses_combined, label='Validation Loss (Combined)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Combined Loss)')
    plt.legend()
    plt.show()

    # Reset model and optimizer for MSE-only training
    model = TempLSTM(input_size, hidden_size, output_size, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train with MSE loss only
    print("\nTraining with MSE loss only:")
    train_losses_mse, val_losses_mse = train_model_mse_only(
        model, train_loader, val_loader, criterion, optimizer, num_epochs
    )

    # Save the MSE-only trained model
    torch.save(model.state_dict(), 'Temperature_forecast_model_mse_only.pt')

    # Plot losses for MSE-only training
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_mse, label='Training Loss (MSE)')
    plt.plot(val_losses_mse, label='Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (MSE Only)')
    plt.legend()
    plt.show()

    # Compare the two training methods
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses_combined, label='Training Loss (Combined)')
    plt.plot(val_losses_combined, label='Validation Loss (Combined)')
    plt.plot(train_losses_mse, label='Training Loss (MSE)')
    plt.plot(val_losses_mse, label='Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Comparison of Training Methods')
    plt.legend()
    plt.show()

    print(f"Number of features used: {input_size}")
    print("\nTraining completed for all methods.")
