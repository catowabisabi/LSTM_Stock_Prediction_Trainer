# -*- coding: utf-8 -*-
"""
Module for defining and building LSTM models using a class-based approach.
"""
import os
import json
import contextlib
import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class LSTMBuilder:
    """
    Builds, trains, and saves LSTM models based on a flexible configuration.
    """
    def __init__(self, model_config, training_config):
        """
        Args:
            model_config (dict): A dictionary defining the model architecture.
            training_config (dict): A dictionary with training parameters.
        """
        self.model_config = model_config
        self.training_config = training_config
        self.model = None

    def build(self, input_shape):
        """
        Builds and compiles a Keras Sequential model from the configuration.

        Args:
            input_shape (tuple): The shape of the input data (window_size, num_features).
        
        Returns:
            A compiled Keras model.
        """
        model = Sequential()
        
        # Add input layer definition
        model.add(LSTM(
            units=self.model_config['layers'][0]['units'],
            return_sequences=self.model_config['layers'][0].get('return_sequences', False),
            input_shape=input_shape
        ))

        # Add subsequent layers
        for layer_config in self.model_config['layers'][1:]:
            layer_type = layer_config['type'].lower()
            
            if layer_type == 'lstm':
                model.add(LSTM(
                    units=layer_config['units'],
                    return_sequences=layer_config.get('return_sequences', False)
                ))
            elif layer_type == 'dense':
                model.add(Dense(
                    units=layer_config['units'],
                    activation=layer_config.get('activation', None)
                ))
            elif layer_type == 'dropout':
                model.add(Dropout(rate=layer_config['rate']))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        optimizer = Adam(learning_rate=self.training_config['learning_rate'])
        model.compile(optimizer=optimizer, loss=self.training_config['loss_function'])
        
        self.model = model
        return self.model

    def save(self, save_path, history, full_config):
        """
        Saves the model, a plot of its training history, and a model card.

        Args:
            save_path (str): The directory to save the model and artifacts.
            history (History): The Keras History object from model.fit().
            full_config (dict): The complete configuration used for the run.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call .build() first.")

        # 1. Create directory
        os.makedirs(save_path, exist_ok=True)
        
        # 2. Save the Keras model file
        model_file_path = os.path.join(save_path, 'model.h5')
        self.model.save(model_file_path)
        print(f"Model file saved to {model_file_path}")

        # 3. Save the training history plot
        plot_path = os.path.join(save_path, 'training_loss.png')
        self._plot_training_history(history, plot_path)
        print(f"Training history plot saved to {plot_path}")

        # 4. Generate and save the model card
        card_path = os.path.join(save_path, 'model_card.md')
        self._generate_model_card(card_path, history, full_config)
        print(f"Model card saved to {card_path}")

    def _plot_training_history(self, history, plot_path):
        """Generates and saves a plot of the training and validation loss."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()

    def _generate_model_card(self, card_path, history, full_config):
        """Creates a markdown file with details about the model and training run."""
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        # Capture model summary
        summary_stream = io.StringIO()
        with contextlib.redirect_stdout(summary_stream):
            self.model.summary()
        summary_string = summary_stream.getvalue()

        card_content = f"""
# Model Card: {self.model_config['name']}

## Model Overview
- **Model Type:** LSTM for Time Series Forecasting
- **Saved At:** {card_path.split(os.path.sep)[-2]}

## Performance
- **Final Training Loss:** `{final_loss:.6f}`
- **Final Validation Loss:** `{final_val_loss:.6f}`

![Training History](training_loss.png)

## Model Architecture
```
{summary_string}
```

## Configuration
The model was trained with the following configuration:
```json
{json.dumps(full_config, indent=2)}
```

## How to Use
```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('{os.path.join(os.path.basename(card_path).replace("model_card.md", ""), "model.h5")}')

# Prepare your input data (X_test) with shape (n_samples, window_size, n_features)
# For this model, window_size={full_config['WINDOW_SIZE']} and n_features=1
# X_test = ... 

# Make predictions
# predictions = model.predict(X_test)
```
"""
        with open(card_path, 'w', encoding='utf-8') as f:
            f.write(card_content) 