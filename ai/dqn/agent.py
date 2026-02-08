"""
DQN Agent
=========

Deep Q-Network agent for oxygen concentrator optimization.

Algorithm: Double DQN with Prioritized Experience Replay

Why DQN for this application:
1. Discrete action space (9 control adjustments)
2. Stable training with experience replay
3. Small model suitable for edge deployment
4. Well-understood exploration-exploitation tradeoff
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from pathlib import Path

from .network import QNetwork, get_model_size
from .replay_buffer import PrioritizedReplayBuffer


@dataclass
class DQNConfig:
    """DQN agent configuration."""
    
    # Environment
    state_dim: int = 14
    action_dim: int = 9
    
    # Network
    hidden_units: Tuple[int, ...] = (64, 64)
    use_dueling: bool = True
    
    # Training
    learning_rate: float = 1e-4
    gamma: float = 0.99          # Discount factor
    tau: float = 0.005           # Soft update rate
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Replay buffer
    buffer_size: int = 100000
    batch_size: int = 64
    min_buffer_size: int = 1000  # Min samples before training
    
    # Training frequency
    train_freq: int = 4          # Steps between training
    target_update_freq: int = 100  # Steps between target updates
    
    # Double DQN
    use_double_dqn: bool = True


class DQNAgent:
    """
    Deep Q-Network agent for control optimization.
    
    Features:
    - Double DQN for reduced overestimation
    - Prioritized Experience Replay
    - Dueling network architecture
    - Soft target network updates
    """
    
    def __init__(self, config: Optional[DQNConfig] = None):
        self.config = config or DQNConfig()
        cfg = self.config
        
        # Networks
        self.q_network = QNetwork(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_units=cfg.hidden_units,
            use_dueling=cfg.use_dueling,
            name='q_network'
        )
        
        self.target_network = QNetwork(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            hidden_units=cfg.hidden_units,
            use_dueling=cfg.use_dueling,
            name='target_network'
        )
        
        # Build networks
        dummy_state = tf.zeros((1, cfg.state_dim))
        self.q_network(dummy_state)
        self.target_network(dummy_state)
        
        # Copy weights to target
        self.target_network.set_weights(self.q_network.get_weights())
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        
        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=cfg.buffer_size,
            state_dim=cfg.state_dim
        )
        
        # Exploration
        self.epsilon = cfg.epsilon_start
        
        # Training state
        self.train_step_count = 0
        self.episode_count = 0
        
        # Metrics
        self.metrics = {
            'loss': [],
            'q_values': [],
            'epsilon': [],
        }
    
    def select_action(self, 
                      state: np.ndarray, 
                      training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            training: If True, use exploration
            
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.config.action_dim)
        
        state = np.expand_dims(state, axis=0).astype(np.float32)
        q_values = self.q_network(state, training=False)
        return int(tf.argmax(q_values[0]).numpy())
    
    def predict(self, state: np.ndarray) -> int:
        """Predict best action (no exploration)."""
        return self.select_action(state, training=False)
    
    def store_experience(self,
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool) -> None:
        """Store experience in replay buffer."""
        self.buffer.add(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        cfg = self.config
        
        # Check if enough samples
        if len(self.buffer) < cfg.min_buffer_size:
            return None
        
        # Check training frequency
        self.train_step_count += 1
        if self.train_step_count % cfg.train_freq != 0:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(cfg.batch_size)
        
        # Convert to tensors
        states = tf.constant(states, dtype=tf.float32)
        actions = tf.constant(actions, dtype=tf.int32)
        rewards = tf.constant(rewards, dtype=tf.float32)
        next_states = tf.constant(next_states, dtype=tf.float32)
        dones = tf.constant(dones, dtype=tf.float32)
        weights = tf.constant(weights, dtype=tf.float32)
        
        # Compute target Q-values
        if cfg.use_double_dqn:
            # Double DQN: use online network to select action
            next_q_online = self.q_network(next_states, training=False)
            next_actions = tf.argmax(next_q_online, axis=1)
            
            # Use target network to evaluate
            next_q_target = self.target_network(next_states, training=False)
            next_q_values = tf.gather(next_q_target, next_actions, axis=1, batch_dims=1)
        else:
            # Standard DQN
            next_q_target = self.target_network(next_states, training=False)
            next_q_values = tf.reduce_max(next_q_target, axis=1)
        
        # Bellman target
        targets = rewards + cfg.gamma * next_q_values * (1 - dones)
        
        # Training step
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            q_action = tf.gather(q_values, actions, axis=1, batch_dims=1)
            
            # TD error
            td_errors = targets - q_action
            
            # Huber loss with importance sampling weights
            loss = tf.reduce_mean(weights * tf.square(td_errors))
        
        # Update network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )
        
        # Update priorities
        self.buffer.update_priorities(indices, td_errors.numpy())
        
        # Soft update target network
        if self.train_step_count % cfg.target_update_freq == 0:
            self._soft_update_target()
        
        # Update metrics
        loss_value = float(loss.numpy())
        self.metrics['loss'].append(loss_value)
        self.metrics['q_values'].append(float(tf.reduce_mean(q_values).numpy()))
        
        return loss_value
    
    def _soft_update_target(self) -> None:
        """Soft update target network weights."""
        tau = self.config.tau
        for target_var, var in zip(
            self.target_network.trainable_variables,
            self.q_network.trainable_variables
        ):
            target_var.assign(tau * var + (1 - tau) * target_var)
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        self.metrics['epsilon'].append(self.epsilon)
    
    def end_episode(self) -> None:
        """Called at end of training episode."""
        self.episode_count += 1
        self.decay_epsilon()
    
    def save(self, path: str) -> None:
        """Save agent weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.q_network.save_weights(str(path / 'q_network.weights.h5'))
        self.target_network.save_weights(str(path / 'target_network.weights.h5'))
        
        # Save config and state
        np.savez(
            str(path / 'agent_state.npz'),
            epsilon=self.epsilon,
            train_step_count=self.train_step_count,
            episode_count=self.episode_count,
        )
        
        print(f"Saved agent to {path}")
    
    def load(self, path: str) -> None:
        """Load agent weights and config."""
        path = Path(path)
        
        self.q_network.load_weights(str(path / 'q_network.weights.h5'))
        self.target_network.load_weights(str(path / 'target_network.weights.h5'))
        
        state = np.load(str(path / 'agent_state.npz'))
        self.epsilon = float(state['epsilon'])
        self.train_step_count = int(state['train_step_count'])
        self.episode_count = int(state['episode_count'])
        
        print(f"Loaded agent from {path}")
    
    def get_model_info(self) -> Dict:
        """Get model size information."""
        return get_model_size(self.q_network)
