"""
Prioritized Experience Replay Buffer
====================================

Implements prioritized experience replay (PER) for DQN training.

Key Features:
- Sum-tree data structure for O(log n) sampling
- Priority based on TD error
- Importance sampling weights for unbiased gradients
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import random


@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class SumTree:
    """
    Sum-tree data structure for efficient prioritized sampling.
    
    Each leaf stores a priority value.
    Parent nodes store sum of children.
    Allows O(log n) sampling proportional to priority.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf node for a given priority sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]
    
    def add(self, priority: float, data) -> None:
        """Add new experience with given priority."""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float) -> None:
        """Update priority of a leaf node."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, any]:
        """Sample leaf based on priority sum."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for DQN.
    
    Priorities are based on TD error:
    p_i = |δ_i| + ε
    
    Sampling probability:
    P(i) = p_i^α / Σ p_k^α
    
    Importance sampling weights:
    w_i = (N * P(i))^(-β) / max(w_j)
    """
    
    PER_E = 0.01  # Small constant to avoid zero priority
    PER_A = 0.6   # Priority exponent (0 = uniform, 1 = full priority)
    PER_B = 0.4   # Initial importance sampling exponent
    PER_B_INCREMENT = 0.001  # Anneal β toward 1
    
    def __init__(self, 
                 capacity: int = 100000,
                 state_dim: int = 14,
                 alpha: float = 0.6,
                 beta: float = 0.4):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum experiences to store
            state_dim: Dimension of state vector
            alpha: Priority exponent
            beta: Initial importance sampling exponent
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.state_dim = state_dim
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = self.PER_B_INCREMENT
        
        # Pre-allocate arrays for batch extraction
        self._states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)
        
        self._max_priority = 1.0
    
    def add(self, 
            state: np.ndarray, 
            action: int, 
            reward: float,
            next_state: np.ndarray,
            done: bool) -> None:
        """Add experience to buffer with max priority."""
        experience = Experience(state, action, reward, next_state, done)
        
        # New experiences get max priority
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        batch_size = min(batch_size, self.tree.n_entries)
        
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        states = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        actions = np.zeros(batch_size, dtype=np.int32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        next_states = np.zeros((batch_size, self.state_dim), dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.bool_)
        
        # Divide priority range into segments
        segment = self.tree.total() / batch_size
        
        # Anneal β
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        p_min = np.min(self.tree.tree[-self.tree.capacity:][
            self.tree.tree[-self.tree.capacity:] > 0
        ]) / self.tree.total()
        max_weight = (p_min * self.tree.n_entries) ** (-self.beta)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, p, exp = self.tree.get(s)
            
            if exp is not None:
                priorities[i] = p
                indices[i] = idx
                states[i] = exp.state
                actions[i] = exp.action
                rewards[i] = exp.reward
                next_states[i] = exp.next_state
                dones[i] = exp.done
        
        # Importance sampling weights
        sampling_probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-self.beta)
        weights /= max_weight  # Normalize
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + self.PER_E) ** self.alpha
            self.tree.update(idx, priority)
            self._max_priority = max(self._max_priority, priority)
    
    def __len__(self) -> int:
        return self.tree.n_entries


class SimpleReplayBuffer:
    """Simple uniform replay buffer (fallback option)."""
    
    def __init__(self, capacity: int = 100000, state_dim: int = 14):
        self.capacity = capacity
        self.state_dim = state_dim
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.write_idx = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.write_idx] = state
        self.actions[self.write_idx] = action
        self.rewards[self.write_idx] = reward
        self.next_states[self.write_idx] = next_state
        self.dones[self.write_idx] = done
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )
    
    def update_priorities(self, indices, td_errors):
        pass  # No-op for uniform buffer
    
    def __len__(self):
        return self.size
