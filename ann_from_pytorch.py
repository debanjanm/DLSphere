import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from base_data_operations import create_classification_dataset
from base_dl_operations import BaseDL

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class PyTorchANN(BaseDL):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001, device='cpu'):
        self.model = ANN(input_size, hidden_size, output_size)
        self.lr = lr
        self.device = device
        self.model.to(self.device)
        super().__init__(self.model)

    def train_model(self, epochs=100):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Call split_data() first.")

        # Convert data to PyTorch tensors
        X_train_tensor = torch.from_numpy(self.X_train).float().to(self.device)
        y_train_tensor = torch.from_numpy(self.y_train).long().to(self.device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Set up the optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        self.model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        print("ANN model trained successfully.")

def main():
    # This main function is just for demonstration
    # You would typically import this class and use it in other files
    X, y = create_classification_dataset()
    ann = PyTorchANN(input_size=X.shape[1], hidden_size=64, output_size=2, lr=0.01)
    ann.load_data()
    ann.split_data()
    ann.train_model(epochs=100)
    ann.evaluate_model()
    ann.predict_sample()

if __name__ == "__main__":
    main()