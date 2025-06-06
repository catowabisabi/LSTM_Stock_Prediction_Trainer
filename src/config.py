# -*- coding: utf-8 -*-
"""
Configuration file for the LSTM Stock Predictor
"""
import datetime
import os

# --- Data Parameters ---
DATA_DIR = 'data'
MODEL_DIR = 'models'
TICKER = 'MSFT.csv'  # Stock ticker symbol, e.g., 'AAPL.csv' or 'MSFT.csv'
TARGET_COL = 'Close'

# --- Preprocessing Parameters ---
# Date range for training and testing
TRAIN_START_DATE = '2016-01-01'
# To select up to the most recent data available in the CSV
TRAIN_END_DATE = '2021-12-31' 
VAL_END_DATE = '2022-12-31'

# The number of past days' data to use for predicting the next day
WINDOW_SIZE = 60 

# --- Model Architecture Definition ---
# Define your model layer by layer.
# Supported types: 'lstm', 'dense', 'dropout'
MODEL_ARCHITECTURE = {
    'name': 'LSTM_v1', # Give your model a unique name
    'layers': [
        {'type': 'lstm', 'units': 64, 'return_sequences': True},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'lstm', 'units': 64, 'return_sequences': False},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'dense', 'units': 32, 'activation': 'relu'},
        {'type': 'dense', 'units': 1} # Output layer
    ]
}

# --- Training Parameters ---
TRAINING_PARAMS = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'loss_function': 'mse' # Mean Squared Error
}

# --- Prediction Parameters ---
# How many future days to predict
PREDICTION_DAYS = 30
# Date from which to start the prediction
PREDICTION_START_DATE = '2023-01-01'

# --- Utility ---
def get_model_save_path():
    """Generates a unique path for saving the model and its card."""
    model_name = MODEL_ARCHITECTURE['name']
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_folder = os.path.join(MODEL_DIR, f"{model_name}_{timestamp}")
    return model_folder 