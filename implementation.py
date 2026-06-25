import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def generate_dummy_data(num_samples, input_size, num_classes):
    X = np.random.rand(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def explain_model(model, input_sample):
    model.eval()
    input_sample = torch.tensor(input_sample, dtype=torch.float32)
    input_sample.requires_grad = True
    output = model(input_sample)
    output_idx = torch.argmax(output, dim=1)
    output_max = output[0, output_idx]
    output_max.backward()
    explanation = input_sample.grad
    return explanation.detach().numpy()

if __name__ == '__main__':
    # Hyperparameters
    input_size = 10
    hidden_size = 5
    output_size = 3
    num_samples = 100
    batch_size = 10
    num_epochs = 5
    learning_rate = 0.01

    # Generate dummy data
    X, y = generate_dummy_data(num_samples, input_size, output_size)
    y_one_hot = np.eye(output_size)[y]
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y_one_hot, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, num_epochs)

    # Explain a single sample
    sample_input = X[0:1]
    explanation = explain_model(model, sample_input)
    print("Input Sample:", sample_input)
    print("Explanation (Gradient):", explanation)