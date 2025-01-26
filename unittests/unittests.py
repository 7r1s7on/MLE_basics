import unittest
import pandas as pd
import os
import sys

# Add the data_process directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_process'))

from data_processing import load_data, split_dataframe, save_dataframe, main


class TestDataDimensions(unittest.TestCase):

    def setUp(self):
        # Define the root directory (MLE_basics)
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Define file paths for train.csv and inference.csv
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.train_file_path = os.path.join(self.data_dir, 'train.csv')
        self.inference_file_path = os.path.join(self.data_dir, 'inference.csv')

        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Load the data and split it
        self.df = load_data()
        self.train_df, self.inference_df = split_dataframe(self.df)

        # Save the dataframes to CSV
        save_dataframe(self.train_df, self.train_file_path)
        save_dataframe(self.inference_df, self.inference_file_path)

    def test_train_csv_dimensions(self):
        # Load the train.csv file
        train_df = pd.read_csv(self.train_file_path)
        
        # Check the dimensions
        self.assertEqual(train_df.shape, (120, 5), "train.csv should have dimensions (120, 5)")

    def test_inference_csv_dimensions(self):
        # Load the inference.csv file
        inference_df = pd.read_csv(self.inference_file_path)
        
        # Check the dimensions
        self.assertEqual(inference_df.shape, (30, 4), "inference.csv should have dimensions (30, 4)")


if __name__ == "__main__":
    unittest.main()