"""
This script trains a simple neural network on the iris dataset.
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Import the configure_logging function from utils
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

def main():
    start_time = time.time()
    logger.info("Starting script execution...")

    # Load dataset
    logger.info("Loading dataset...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    relative_path = os.path.join(parent_dir, 'data', 'train.csv')
    absolute_path = os.path.abspath(relative_path)

    df = pd.read_csv(absolute_path)
    logger.info(f"Dataset loaded. Size: {len(df)} samples.")

    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split dataset
    logger.info("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {len(X_train)} samples.")
    logger.info(f"Testing set size: {len(X_test)} samples.")

    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    # Initialize model, loss function, and optimizer
    logger.info("Initializing model...")
    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    logger.info("Starting training...")
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    logger.info("Training completed.")

    # Evaluate the model
    logger.info("Evaluating model on test set...")
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        logger.info(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Save the model
    logger.info("Saving model...")
    model_dir = os.path.join(parent_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}.")

    # Log total time spent
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total time spent: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()