"""
Q-Network Architecture
======================

Neural network for DQN agent, optimized for TensorFlow Lite conversion.

Design Choices:
- Small footprint (< 50KB TFLite)
- No batch normalization (stateful layers problematic for TFLite)
- ReLU6 activation (quantization-friendly)
- Discrete action space output
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


def create_q_network(
    state_dim: int = 14,
    action_dim: int = 9,
    hidden_units: Tuple[int, ...] = (64, 64, 32),
    activation: str = 'relu6',
    name: str = 'q_network'
) -> keras.Model:
    """
    Create Q-Network for DQN agent.
    
    Architecture:
        Input (state_dim) -> Dense(64) -> Dense(64) -> Dense(32) -> Dense(action_dim)
    
    Args:
        state_dim: Dimension of state vector
        action_dim: Number of discrete actions
        hidden_units: Tuple of hidden layer sizes
        activation: Activation function (relu6 for quantization)
        name: Model name
        
    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=(state_dim,), name='state_input')
    
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units, 
            activation=activation,
            name=f'hidden_{i}'
        )(x)
    
    # Q-value output (no activation - linear)
    q_values = layers.Dense(
        action_dim, 
        activation=None,
        name='q_values'
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=q_values, name=name)
    
    return model


class QNetwork(keras.Model):
    """
    Q-Network with dueling architecture option.
    
    Dueling DQN separates state value and action advantage:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    
    This helps learning when actions have similar values.
    """
    
    def __init__(self,
                 state_dim: int = 14,
                 action_dim: int = 9,
                 hidden_units: Tuple[int, ...] = (64, 64),
                 use_dueling: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_dueling = use_dueling
        
        # Shared layers
        self.hidden_layers = [
            layers.Dense(units, activation='relu6', name=f'hidden_{i}')
            for i, units in enumerate(hidden_units)
        ]
        
        if use_dueling:
            # Value stream
            self.value_hidden = layers.Dense(32, activation='relu6', name='value_hidden')
            self.value_output = layers.Dense(1, activation=None, name='value')
            
            # Advantage stream
            self.advantage_hidden = layers.Dense(32, activation='relu6', name='advantage_hidden')
            self.advantage_output = layers.Dense(action_dim, activation=None, name='advantage')
        else:
            # Standard Q output
            self.q_output = layers.Dense(action_dim, activation=None, name='q_values')
    
    def call(self, state: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass."""
        x = state
        
        # Shared hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        if self.use_dueling:
            # Dueling architecture
            value = self.value_hidden(x)
            value = self.value_output(value)
            
            advantage = self.advantage_hidden(x)
            advantage = self.advantage_output(advantage)
            
            # Combine: Q = V + A - mean(A)
            q_values = value + advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)
        else:
            q_values = self.q_output(x)
        
        return q_values
    
    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'use_dueling': self.use_dueling,
        }


def convert_to_tflite(
    model: keras.Model,
    output_path: str,
    quantize: bool = True,
    representative_dataset=None
) -> bytes:
    """
    Convert Keras model to TensorFlow Lite format.
    
    Args:
        model: Trained Keras model
        output_path: Path to save .tflite file
        quantize: Apply int8 quantization (reduces size ~4x)
        representative_dataset: Generator for quantization calibration
        
    Returns:
        TFLite model as bytes
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset is not None:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save to file
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved TFLite model to {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.1f} KB")
    
    return tflite_model


def get_model_size(model: keras.Model) -> dict:
    """Get model size statistics."""
    # Count parameters
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    
    return {
        'trainable_params': trainable,
        'non_trainable_params': non_trainable,
        'total_params': trainable + non_trainable,
        'estimated_size_kb': (trainable + non_trainable) * 4 / 1024,  # float32
    }
