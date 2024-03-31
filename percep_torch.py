import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Function to load and preprocess the Isolet dataset
def load_dataset(batch_size=32):
    # Load datasets
    df_train = pd.read_csv('isolet1+2+3+4.data', header=None)
    df_test = pd.read_csv('isolet5.data', header=None)

    # Split into features and labels, and preprocess
    scaler = StandardScaler()
    encoder = OneHotEncoder(categories='auto')
    
    X_train = scaler.fit_transform(df_train.iloc[:, :-1])
    Y_train = encoder.fit_transform(df_train.iloc[:, -1].values.reshape(-1, 1) - 1).toarray()
    X_test = scaler.transform(df_test.iloc[:, :-1])
    Y_test = encoder.transform(df_test.iloc[:, -1].values.reshape(-1, 1) - 1).toarray()

    # Convert to tensors and create data loaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Training function
def train_model(model, criterion, optimizer, train_loader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(targets.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')

# Load dataset
train_loader, test_loader = load_dataset()

# Define the neural network
input_size = 617
layer_sizes = [64]
output_size = 26

model = nn.Sequential(
    nn.Linear(input_size, layer_sizes[0]),
    nn.ReLU(),
    #nn.Linear(layer_sizes[0], layer_sizes[1]),
    #nn.ReLU(),
    nn.Linear(layer_sizes[0], output_size)
)

# Set up criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate the model
train_model(model, criterion, optimizer, train_loader, epochs=10)
evaluate_model(model, test_loader)

