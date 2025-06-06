# -*- coding: utf-8 -*-
"""
Module for loading and preprocessing stock data.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(data_dir, ticker, target_col):
    """
    Loads stock data from a CSV file.
    
    Args:
        data_dir (str): The directory where the data is stored.
        ticker (str): The filename of the stock data CSV.
        target_col (str): The name of the column to be used as the target variable.

    Returns:
        pd.DataFrame: A DataFrame with 'Date' as the index and the target column.
    """
    csv_path = os.path.join(data_dir, ticker)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    df = df[['Date', target_col]]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def preprocess_data(df, window_size):
    """
    Preprocesses the data by scaling and creating windowed datasets.

    Args:
        df (pd.DataFrame): The input data.
        window_size (int): The size of the sliding window.

    Returns:
        tuple: A tuple containing:
            - np.array: The windowed input data (X).
            - np.array: The target data (y).
            - MinMaxScaler: The scaler used for the data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # Reshape X to be [samples, time steps, features] which is required for LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def split_data(df, train_end, val_end):
    """
    Splits the data into training, validation, and test sets based on dates.
    
    Args:
        df (pd.DataFrame): The input dataframe with a DatetimeIndex.
        train_end (str): The end date for the training set (e.g., 'YYYY-MM-DD').
        val_end (str): The end date for the validation set (e.g., 'YYYY-MM-DD').
        
    Returns:
        tuple: A tuple containing (train_df, validation_df, test_df).
    """
    train_end_date = pd.to_datetime(train_end)
    val_end_date = pd.to_datetime(val_end)

    train_df = df.loc[:train_end_date]
    validation_df = df.loc[train_end_date:val_end_date]
    test_df = df.loc[val_end_date:]
    
    return train_df, validation_df, test_df 