# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# # Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import configure_logging

def load_data():
    df = load_iris()
    df = pd.DataFrame(data=np.c_[df['data'], df['target']], columns=df['feature_names'] + ['target'])
    return df

def split_dataframe(df, train_ratio=0.8):
    train_df, inference_df = train_test_split(df, train_size=train_ratio, random_state=42)
    inference_df = inference_df.drop(columns='target')
    return train_df, inference_df

def save_dataframe(df, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

def main():
    logger.info("Starting the script...")
    configure_logging()

    # Split the DataFrame
    logger.info("Loading data...")
    df = load_data()
    logger.info("Splitting data...")
    train_df, inference_df = split_dataframe(df)

    # Save the DataFrames
    logger.info("Saving data...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    train_file_path = os.path.join(parent_dir, 'data', 'train.csv')
    inference_file_path = os.path.join(parent_dir, 'data', 'inference.csv')

    save_dataframe(train_df, train_file_path)
    save_dataframe(inference_df, inference_file_path)
    logger.info("Data saved successfully!!!")

if __name__ == "__main__":
    main()
