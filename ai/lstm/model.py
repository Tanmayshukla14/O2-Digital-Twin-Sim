"""
LSTM Health Predictor
=====================

LSTM-based predictive maintenance model for oxygen concentrator.

Purpose:
- Predict remaining useful life (RUL)
- Detect anomalies in operating patterns
- Estimate component health scores

Input: Sliding window of sensor readings (60 timesteps)
Output: Health score [0-1] and anomaly flag

Why LSTM:
1. Captures temporal patterns in degradation
2. Handles variable-length sequences
3. Good at detecting gradual trends
4. Small enough for edge deployment
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from pathlib import Path


@dataclass
class LSTMConfig:
    """LSTM model configuration."""
    
    # Input
    sequence_length: int = 60         # Input window (60 seconds at 1Hz)
    feature_dim: int = 8              # Number of input features
    
    # Architecture
    lstm_units: Tuple[int, ...] = (32, 16)  # LSTM layer sizes
    dense_units: Tuple[int, ...] = (16,)    # Dense layer sizes
    dropout_rate: float = 0.2
    
    # Output
    output_dim: int = 2               # [health_score, anomaly_score]
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    
    # Thresholds
    anomaly_threshold: float = 0.7    # Score above this = anomaly
    health_warning_threshold: float = 0.8  # Health below this = warning


@dataclass
class HealthPrediction:
    """Health prediction output."""
    health_score: float            # 0 = failed, 1 = healthy
    anomaly_score: float           # 0 = normal, 1 = anomaly
    is_anomaly: bool
    health_warning: bool
    confidence: float
    feature_contributions: Dict[str, float]


class LSTMHealthPredictor:
    """
    LSTM-based predictive maintenance model.
    
    Features monitored:
    1. Motor current (signature analysis)
    2. Motor temperature (thermal trend)
    3. Compressor pressure (performance)
    4. Flow rate (efficiency)
    5. O2 purity (PSA health)
    6. Cycle count (wear)
    7. Vibration (if available)
    8. Power consumption (efficiency)
    
    Training data: Operating logs with labeled health states
    """
    
    FEATURE_NAMES = [
        'motor_current',
        'motor_temperature', 
        'pressure',
        'flow_rate',
        'o2_purity',
        'cycle_count_normalized',
        'power_consumption',
        'pressure_variance'
    ]
    
    def __init__(self, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.model = self._build_model()
        
        # Feature normalization
        self.feature_means = np.zeros(self.config.feature_dim)
        self.feature_stds = np.ones(self.config.feature_dim)
        
        # Sliding window buffer
        self.buffer = np.zeros((self.config.sequence_length, self.config.feature_dim))
        self.buffer_idx = 0
        self.buffer_filled = False
    
    def _build_model(self) -> keras.Model:
        """Build LSTM model."""
        cfg = self.config
        
        inputs = keras.Input(
            shape=(cfg.sequence_length, cfg.feature_dim),
            name='sequence_input'
        )
        
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(cfg.lstm_units):
            return_sequences = i < len(cfg.lstm_units) - 1
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=cfg.dropout_rate if i > 0 else 0,
                name=f'lstm_{i}'
            )(x)
        
        # Dense layers
        for i, units in enumerate(cfg.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
            x = layers.Dropout(cfg.dropout_rate)(x)
        
        # Output: [health_score, anomaly_score]
        # Both sigmoid for [0, 1] range
        outputs = layers.Dense(
            cfg.output_dim, 
            activation='sigmoid',
            name='predictions'
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='health_predictor')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def reset(self) -> None:
        """Reset sliding window buffer."""
        self.buffer = np.zeros((self.config.sequence_length, self.config.feature_dim))
        self.buffer_idx = 0
        self.buffer_filled = False
    
    def update(self, features: Dict[str, float]) -> None:
        """
        Add new measurement to sliding window.
        
        Args:
            features: Dict with feature values
        """
        # Extract features in order
        feature_vector = np.array([
            features.get('motor_current', 0),
            features.get('motor_temperature', 25),
            features.get('pressure', 2.0),
            features.get('flow_rate', 3.0),
            features.get('o2_purity', 0.93),
            features.get('cycle_count_normalized', 0),
            features.get('power_consumption', 100),
            features.get('pressure_variance', 0),
        ])
        
        # Normalize
        feature_vector = (feature_vector - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Add to buffer (circular)
        self.buffer[self.buffer_idx] = feature_vector
        self.buffer_idx = (self.buffer_idx + 1) % self.config.sequence_length
        
        if self.buffer_idx == 0:
            self.buffer_filled = True
    
    def predict(self) -> Optional[HealthPrediction]:
        """
        Predict health from current buffer.
        
        Returns:
            HealthPrediction or None if buffer not filled
        """
        if not self.buffer_filled:
            return None
        
        # Reorder buffer to time order
        sequence = np.roll(self.buffer, -self.buffer_idx, axis=0)
        sequence = np.expand_dims(sequence, axis=0)
        
        # Predict
        prediction = self.model.predict(sequence, verbose=0)[0]
        health_score = float(prediction[0])
        anomaly_score = float(prediction[1])
        
        # Compute feature contributions (simplified gradient-based)
        contributions = self._compute_contributions(sequence)
        
        return HealthPrediction(
            health_score=health_score,
            anomaly_score=anomaly_score,
            is_anomaly=anomaly_score > self.config.anomaly_threshold,
            health_warning=health_score < self.config.health_warning_threshold,
            confidence=1.0 - abs(anomaly_score - 0.5) * 2,  # Higher near 0 or 1
            feature_contributions=contributions
        )
    
    def _compute_contributions(self, sequence: np.ndarray) -> Dict[str, float]:
        """Compute feature contributions (simplified)."""
        # Use variance of each feature as proxy for contribution
        variances = np.var(sequence[0], axis=0)
        total = np.sum(variances) + 1e-8
        
        contributions = {}
        for i, name in enumerate(self.FEATURE_NAMES):
            contributions[name] = float(variances[i] / total)
        
        return contributions
    
    def fit_normalizer(self, data: np.ndarray) -> None:
        """
        Fit feature normalizer on training data.
        
        Args:
            data: Array of shape (n_samples, sequence_length, feature_dim)
        """
        # Flatten to compute statistics
        flat = data.reshape(-1, self.config.feature_dim)
        self.feature_means = np.mean(flat, axis=0)
        self.feature_stds = np.std(flat, axis=0)
    
    def save(self, path: str) -> None:
        """Save model and normalizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(path / 'lstm_model.h5'))
        np.savez(
            str(path / 'normalizer.npz'),
            means=self.feature_means,
            stds=self.feature_stds
        )
        print(f"Saved LSTM model to {path}")
    
    def load(self, path: str) -> None:
        """Load model and normalizer."""
        path = Path(path)
        
        self.model = keras.models.load_model(str(path / 'lstm_model.h5'))
        
        normalizer = np.load(str(path / 'normalizer.npz'))
        self.feature_means = normalizer['means']
        self.feature_stds = normalizer['stds']
        
        print(f"Loaded LSTM model from {path}")
    
    def get_model_summary(self) -> str:
        """Get model summary string."""
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


def create_synthetic_training_data(
    n_samples: int = 1000,
    sequence_length: int = 60,
    feature_dim: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic training data for initial model development.
    
    Simulates:
    - Normal operation patterns
    - Gradual degradation
    - Anomaly events
    
    Returns:
        X: Input sequences (n_samples, sequence_length, feature_dim)
        y: Labels (n_samples, 2) [health_score, anomaly_score]
    """
    X = np.zeros((n_samples, sequence_length, feature_dim))
    y = np.zeros((n_samples, 2))
    
    for i in range(n_samples):
        # Randomly choose scenario
        scenario = np.random.choice(['normal', 'degrading', 'anomaly'], p=[0.6, 0.3, 0.1])
        
        if scenario == 'normal':
            # Normal operation with noise
            base = np.array([5.0, 45.0, 2.0, 3.0, 0.93, 0.0, 100, 0.01])
            noise = np.random.normal(0, 0.1, (sequence_length, feature_dim))
            X[i] = base + noise
            y[i] = [1.0, 0.0]  # Healthy, no anomaly
            
        elif scenario == 'degrading':
            # Gradual degradation trend
            health = np.random.uniform(0.3, 0.9)
            base = np.array([5.0, 45.0, 2.0, 3.0, 0.93, 0.0, 100, 0.01])
            
            # Add trend
            trend = np.linspace(0, (1 - health) * 0.5, sequence_length)
            X[i, :, 0] = base[0] + trend + np.random.normal(0, 0.1, sequence_length)  # Current increase
            X[i, :, 1] = base[1] + trend * 20 + np.random.normal(0, 1, sequence_length)  # Temp increase
            X[i, :, 2:] = base[2:] + np.random.normal(0, 0.1, (sequence_length, feature_dim - 2))
            
            y[i] = [health, 0.2]
            
        else:  # anomaly
            # Sudden anomaly pattern
            base = np.array([5.0, 45.0, 2.0, 3.0, 0.93, 0.0, 100, 0.01])
            X[i] = base + np.random.normal(0, 0.1, (sequence_length, feature_dim))
            
            # Insert anomaly at random point
            anomaly_start = np.random.randint(30, 50)
            X[i, anomaly_start:, 0] *= 1.5  # Current spike
            X[i, anomaly_start:, 7] *= 5    # Pressure variance spike
            
            y[i] = [0.7, 0.9]  # Reduced health, anomaly
    
    return X, y
