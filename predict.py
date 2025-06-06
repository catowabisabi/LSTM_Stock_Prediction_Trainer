# -*- coding: utf-8 -*-
"""
Main script for making predictions with a trained LSTM model.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from src import config
from src.data_loader import load_data

def find_latest_model_path(model_dir):
    """Finds the path to the model.h5 file in the latest model folder."""
    if not os.path.exists(model_dir):
        return None
        
    subfolders = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    
    if not subfolders:
        return None
        
    latest_folder = max(subfolders, key=os.path.getctime)
    model_path = os.path.join(latest_folder, 'model.h5')
    
    return model_path if os.path.exists(model_path) else None

def plot_predictions(true_data, predicted_data, model_folder):
    """Saves a plot of the true vs. predicted stock prices."""
    plt.figure(figsize=(15, 7))
    plt.plot(true_data.index, true_data, color='blue', label='Actual Price')
    plt.plot(predicted_data.index, predicted_data, color='red', label='Predicted Price', linestyle='--')
    plt.title(f'Stock Price Prediction ({config.TICKER})')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(model_folder, 'prediction_vs_actual.png')
    plt.savefig(plot_filename)
    print(f"Prediction plot saved to {plot_filename}")
    plt.close()

def predict(model_folder_path=None):
    """
    Main prediction pipeline.
    """
    # --- 1. Load Model ---
    if model_folder_path is None:
        print("No model folder provided, searching for the latest model...")
        model_h5_path = find_latest_model_path(config.MODEL_DIR)
        if model_h5_path is None:
            raise FileNotFoundError(f"No 'model.h5' found in any subfolder of '{config.MODEL_DIR}'. Please train a model first.")
        model_folder_path = os.path.dirname(model_h5_path)
    else:
        model_h5_path = os.path.join(model_folder_path, 'model.h5')

    print(f"Loading model from: {model_h5_path}")
    if not os.path.exists(model_h5_path):
        raise FileNotFoundError(f"Model file not found at {model_h5_path}")
    model = load_model(model_h5_path)

    # --- 2. Load Data ---
    print("Loading data for prediction...")
    df = load_data(config.DATA_DIR, config.TICKER, config.TARGET_COL)

    # We need the last `window_size` days from the training/validation data to start predicting
    prediction_start_date = pd.to_datetime(config.PREDICTION_START_DATE)
    required_start_date = prediction_start_date - pd.DateOffset(days=config.WINDOW_SIZE * 2) # Fetch more to be safe
    
    inputs_df = df.loc[required_start_date:prediction_start_date].copy()
    
    # --- 3. Preprocess Data ---
    scaler = MinMaxScaler(feature_range=(0,1))
    # Note: We fit the scaler on the same data range the model was trained on for consistency
    training_data_for_scaling = df.loc[:config.TRAIN_END_DATE]
    scaler.fit(training_data_for_scaling[[config.TARGET_COL]])
    
    scaled_inputs = scaler.transform(inputs_df[[config.TARGET_COL]])

    # Take the last `window_size` points from the available data
    X_test = []
    X_test.append(scaled_inputs[-config.WINDOW_SIZE:, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # --- 4. Make Predictions ---
    print("Making predictions...")
    predicted_price_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    # --- 5. Display and Plot Results ---
    prediction_date = df.index[df.index.searchsorted(prediction_start_date)]
    print(f"Predicted price for {prediction_date.date()}: {predicted_price[0][0]:.2f}")

    # For plotting, let's get a range of actual data to compare against
    actual_data_to_plot = df.loc[prediction_start_date - pd.DateOffset(months=2):prediction_start_date]
    predicted_df = pd.DataFrame(predicted_price, 
                                index=[prediction_date], 
                                columns=['Predicted'])

    plot_predictions(actual_data_to_plot[config.TARGET_COL], predicted_df['Predicted'], model_folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str, help='Path to the model folder (e.g., models/LSTM_v1_20230101-120000)')
    args = parser.parse_args()
    
    predict(args.model_folder) 