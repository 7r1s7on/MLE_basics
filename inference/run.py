"""
This script performs inference using a trained neural network on the iris dataset.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        """
        Small neural net because of the small dataset
        """
        self.fc1 = nn.Linear(4, 8)  # Input layer of 4 features, hidden layer of 8 nodes
        self.fc2 = nn.Linear(8, 16)  # Hidden layer of 16 nodes
        self.fc3 = nn.Linear(16, 3)  # Output layer of 3 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    try:
        logger.info("Loading model...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
        model = IrisNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_data(data_path):
    """
    Load and preprocess the inference dataset.
    """
    try:
        logger.info("Loading inference dataset...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Inference dataset not found at {data_path}.")
        
        df = pd.read_csv(data_path)
        logger.info(f"Inference dataset loaded. Size: {len(df)} samples.")

        # Check if the dataset has the required features
        required_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        if not all(feature in df.columns for feature in required_features):
            raise ValueError(f"Inference dataset must contain the following features: {required_features}")

        # Handle case where target column is missing
        X = df.drop('target', axis=1) if 'target' in df.columns else df

        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        return X_tensor, df
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def make_predictions(model, X):
    """
    Make predictions using the trained model.
    """
    try:
        logger.info("Making predictions...")
        with torch.no_grad():
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def save_results(df, predictions, output_path):
    """
    Save the predictions to a CSV file.
    """
    try:
        logger.info("Saving results...")
        df['predicted_target'] = predictions
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}.")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def main():
    try:
        # Paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        model_path = os.path.join(parent_dir, 'models', 'model.pth')
        inference_data_path = os.path.join(parent_dir, 'data', 'inference.csv')
        output_path = os.path.join(parent_dir, 'results', 'predictions.csv')

        # Ensure the results directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load the model
        model = load_model(model_path)

        # Preprocess the inference data
        X, df = preprocess_data(inference_data_path)

        # Make predictions
        predictions = make_predictions(model, X)

        # Save the results
        save_results(df, predictions, output_path)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()