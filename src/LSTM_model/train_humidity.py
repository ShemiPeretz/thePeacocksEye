import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import HumidityLSTM
from train_temp import train_model_with_cosine_loss, train_model_mse_only

if __name__ == "__main__":
    # Load preprocessed data
    X_train = torch.load('X_train_Relative humidity (%).pt')
    X_test = torch.load('X_test_Relative humidity (%).pt')
    y_train = torch.load('y_train_Relative humidity (%).pt')
    y_test = torch.load('y_test_Relative humidity (%).pt')

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
    dropout = 0.2
    alpha = 2  # Reduced weight for cosine similarity loss

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Check the shape of the data
    X_sample, y_sample = next(iter(train_loader))
    print(f"Input shape: {X_sample.shape}, Output shape: {y_sample.shape}")

    # Define model parameters
    input_size = 421
    hidden_size = 421
    output_size = 1
    num_layers = 2

    # Initialize the HumidityLSTM model
    model = HumidityLSTM(input_size, hidden_size, output_size, num_layers, dropout)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train with combined loss (MSE + Cosine Similarity)
    print("Training with combined loss (MSE + Cosine Similarity):")
    train_losses_combined, val_losses_combined = train_model_with_cosine_loss(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, alpha
    )

    # Save the trained model
    torch.save(model.state_dict(), 'Humidity_forecast_model_combined_loss.pt')

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
    model = HumidityLSTM(input_size, hidden_size, output_size, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train with MSE loss only
    print("\nTraining with MSE loss only:")
    train_losses_mse, val_losses_mse = train_model_mse_only(
        model, train_loader, val_loader, criterion, optimizer, num_epochs
    )

    # Save the MSE-only trained model
    torch.save(model.state_dict(), 'Humidity_forecast_model_mse_only.pt')

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
