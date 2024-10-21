import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from base_data_operations import create_classification_dataset
import torch

class BaseDL:
    def __init__(self, model):
        self.model = model
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        X, y = create_classification_dataset()
        self.X = X.values
        self.y = y.values
        print("Data loaded successfully.")
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")

    def split_data(self, test_size=0.2, random_state=42):
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print("Data split successfully.")
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")

    def train_model(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Call split_data() first.")
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully.")

    def evaluate_model(self):
            if self.X_test is None or self.y_test is None:
                raise ValueError("Data not split. Call split_data() first.")

            # Convert the test data to PyTorch tensors and move them to the device
            X_test_tensor = torch.from_numpy(self.X_test).float().to(self.device)
            y_test_tensor = torch.from_numpy(self.y_test).long().to(self.device)

            # Evaluate the model
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_test_tensor)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = accuracy_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy())
                print(f"Accuracy: {accuracy:.2f}")

                print("\nClassification Report:")
                print(classification_report(y_test_tensor.cpu().numpy(), predicted.cpu().numpy()))

    def predict_sample(self, sample_index=0):
            if self.X_test is None or self.y_test is None:
                raise ValueError("Data not split. Call split_data() first.")

            # Convert the sample to a PyTorch tensor and move it to the device
            sample = torch.from_numpy(self.X_test[sample_index]).float().unsqueeze(0).to(self.device)

            # Make a prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(sample)
                predicted_class = torch.argmax(prediction, dim=1).item()

            print(f"\nPrediction for sample at index {sample_index}: {predicted_class}")
            print(f"Actual value: {self.y_test[sample_index]}")