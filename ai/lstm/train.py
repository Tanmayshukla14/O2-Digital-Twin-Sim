"""
LSTM Training Script
====================

Training pipeline for LSTM health predictor.

Usage:
    python -m ai.lstm.train --epochs 100 --data-dir data/
"""

import argparse
import numpy as np
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras

from .model import (
    LSTMHealthPredictor, 
    LSTMConfig,
    create_synthetic_training_data
)


def train_lstm(
    data_dir: str = 'data',
    output_dir: str = 'checkpoints/lstm',
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    use_synthetic: bool = True,
    seed: int = 42
) -> LSTMHealthPredictor:
    """
    Train LSTM health predictor.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory for checkpoints
        epochs: Training epochs
        batch_size: Batch size
        validation_split: Validation split ratio
        use_synthetic: Use synthetic data if no real data
        seed: Random seed
        
    Returns:
        Trained model
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load or generate data
    data_path = Path(data_dir)
    
    if (data_path / 'X_train.npy').exists() and not use_synthetic:
        print("Loading training data...")
        X = np.load(data_path / 'X_train.npy')
        y = np.load(data_path / 'y_train.npy')
    else:
        print("Generating synthetic training data...")
        X, y = create_synthetic_training_data(n_samples=5000)
        
        # Save for reproducibility
        data_path.mkdir(parents=True, exist_ok=True)
        np.save(data_path / 'X_train.npy', X)
        np.save(data_path / 'y_train.npy', y)
    
    print(f"Training data: {X.shape}")
    print(f"Labels: {y.shape}")
    
    # Create model
    config = LSTMConfig(
        sequence_length=X.shape[1],
        feature_dim=X.shape[2],
        lstm_units=(32, 16),
        dense_units=(16,),
        dropout_rate=0.2,
    )
    
    model = LSTMHealthPredictor(config)
    model.fit_normalizer(X)
    
    # Normalize data
    X_normalized = (X - model.feature_means) / (model.feature_stds + 1e-8)
    
    # Split
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint(
            str(output_path / 'best_model.h5'),
            save_best_only=True
        ),
    ]
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    history = model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    model.save(str(output_path))
    
    # Save history
    with open(output_path / 'history.json', 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    
    # Evaluation
    print("\nEvaluation:")
    val_loss, val_mae = model.model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train LSTM health predictor')
    parser.add_argument('--data-dir', type=str, default='data/lstm')
    parser.add_argument('--output-dir', type=str, default='checkpoints/lstm')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    train_lstm(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_synthetic=args.synthetic,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
