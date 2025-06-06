# -*- coding: utf-8 -*-
"""
Main script for training the LSTM model.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src import config
from src.data_loader import load_data, preprocess_data, split_data
from src.model import LSTMBuilder

def plot_training_history(history, model_path):
    """Saves a plot of the training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plot_filename = os.path.splitext(model_path)[0] + '_loss.png'
    plt.savefig(plot_filename)
    print(f"Training history plot saved to {plot_filename}")
    plt.close()

def get_full_config():
    """Returns a dictionary of all relevant configurations."""
    return {
        'TICKER': config.TICKER,
        'TARGET_COL': config.TARGET_COL,
        'TRAIN_START_DATE': config.TRAIN_START_DATE,
        'TRAIN_END_DATE': config.TRAIN_END_DATE,
        'VAL_END_DATE': config.VAL_END_DATE,
        'WINDOW_SIZE': config.WINDOW_SIZE,
        'MODEL_ARCHITECTURE': config.MODEL_ARCHITECTURE,
        'TRAINING_PARAMS': config.TRAINING_PARAMS
    }

def train():
    """
    The main training pipeline using the LSTMBuilder.
    """
    # --- 1. Load Data ---
    print("Loading data...")
    df = load_data(config.DATA_DIR, config.TICKER, config.TARGET_COL)
    
    # --- 2. Split Data ---
    print("Splitting data into train, validation, and test sets...")
    train_df, val_df, _ = split_data(df, config.TRAIN_END_DATE, config.VAL_END_DATE)
    
    # --- 3. Preprocess Data ---
    print("Preprocessing data...")
    X_train, y_train, scaler = preprocess_data(train_df, config.WINDOW_SIZE)
    
    scaled_val_data = scaler.transform(val_df)
    X_val, y_val = [], []
    for i in range(config.WINDOW_SIZE, len(scaled_val_data)):
        X_val.append(scaled_val_data[i-config.WINDOW_SIZE:i, 0])
        y_val.append(scaled_val_data[i, 0])
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError("Training or validation set is empty. Check date ranges and data.")

    # --- 4. Build Model ---
    print("Building model...")
    builder = LSTMBuilder(config.MODEL_ARCHITECTURE, config.TRAINING_PARAMS)
    input_shape = (X_train.shape[1], 1)
    model = builder.build(input_shape)
    model.summary()

    # --- 5. Train Model ---
    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=config.TRAINING_PARAMS['batch_size'],
        epochs=config.TRAINING_PARAMS['epochs'],
        validation_data=(X_val, y_val)
    )

    # --- 6. Save Model and Model Card ---
    print("Saving model and generating model card...")
    save_path = config.get_model_save_path()
    full_config = get_full_config()
    builder.save(save_path, history, full_config)
    print(f"All artifacts saved in: {save_path}")

if __name__ == '__main__':
    train() 