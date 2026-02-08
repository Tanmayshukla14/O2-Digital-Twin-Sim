"""
DQN Training Script
===================

Training pipeline for DQN agent with logging and checkpointing.

Usage:
    python -m ai.dqn.train --episodes 1000 --checkpoint-dir checkpoints/
"""

import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json

# TensorFlow logging level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from .agent import DQNAgent, DQNConfig
from .network import convert_to_tflite

# Import simulator (relative import)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from simulator.environment import OxygenConcentratorEnv, EnvironmentConfig


class TrainingLogger:
    """Training metrics logger."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_spo2: List[float] = []
        self.losses: List[float] = []
        
        # TensorBoard
        self.writer = tf.summary.create_file_writer(str(self.log_dir / 'tensorboard'))
    
    def log_episode(self, 
                    episode: int,
                    reward: float,
                    length: int,
                    mean_spo2: float,
                    epsilon: float,
                    loss: float) -> None:
        """Log episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_spo2.append(mean_spo2)
        if loss is not None:
            self.losses.append(loss)
        
        with self.writer.as_default():
            tf.summary.scalar('episode/reward', reward, step=episode)
            tf.summary.scalar('episode/length', length, step=episode)
            tf.summary.scalar('episode/mean_spo2', mean_spo2, step=episode)
            tf.summary.scalar('training/epsilon', epsilon, step=episode)
            if loss is not None:
                tf.summary.scalar('training/loss', loss, step=episode)
    
    def get_stats(self, window: int = 100) -> Dict:
        """Get recent statistics."""
        recent_rewards = self.episode_rewards[-window:]
        recent_spo2 = self.episode_spo2[-window:]
        
        return {
            'mean_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'std_reward': np.std(recent_rewards) if recent_rewards else 0,
            'mean_spo2': np.mean(recent_spo2) if recent_spo2 else 0,
            'total_episodes': len(self.episode_rewards),
        }
    
    def save(self) -> None:
        """Save logs to file."""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_spo2': self.episode_spo2,
            'losses': self.losses,
        }
        with open(self.log_dir / 'training_log.json', 'w') as f:
            json.dump(data, f)


def train_dqn(
    episodes: int = 1000,
    max_steps: int = 3000,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
    checkpoint_freq: int = 100,
    eval_freq: int = 50,
    seed: int = 42
) -> DQNAgent:
    """
    Train DQN agent.
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        checkpoint_dir: Directory for saving checkpoints
        log_dir: Directory for training logs
        checkpoint_freq: Episodes between checkpoints
        eval_freq: Episodes between evaluation
        seed: Random seed
        
    Returns:
        Trained DQN agent
    """
    # Set seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Create environment
    env_config = EnvironmentConfig(
        episode_duration=300.0,  # 5 minutes
        randomize_patient=True,
        randomize_environment=True,
    )
    env = OxygenConcentratorEnv(env_config)
    
    # Create agent
    agent_config = DQNConfig(
        state_dim=14,
        action_dim=9,
        hidden_units=(64, 64),
        use_dueling=True,
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=100000,
        batch_size=64,
        min_buffer_size=1000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )
    agent = DQNAgent(agent_config)
    
    # Create logger
    logger = TrainingLogger(log_dir)
    
    # Checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting DQN training: {episodes} episodes")
    print(f"Model info: {agent.get_model_info()}")
    print("-" * 60)
    
    best_reward = float('-inf')
    start_time = time.time()
    
    for episode in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_spo2 = []
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Select action
            action = agent.select_action(obs, training=True)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Train
            loss = agent.train_step()
            
            # Track metrics
            episode_reward += reward
            episode_spo2.append(info['spo2'])
            
            obs = next_obs
            step += 1
        
        # End of episode
        agent.end_episode()
        
        # Log metrics
        mean_spo2 = np.mean(episode_spo2) if episode_spo2 else 0
        avg_loss = np.mean(agent.metrics['loss'][-100:]) if agent.metrics['loss'] else None
        
        logger.log_episode(
            episode=episode,
            reward=episode_reward,
            length=step,
            mean_spo2=mean_spo2,
            epsilon=agent.epsilon,
            loss=avg_loss
        )
        
        # Print progress
        if episode % 10 == 0:
            stats = logger.get_stats()
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{episodes} | "
                  f"Reward: {episode_reward:.1f} (avg: {stats['mean_reward']:.1f}) | "
                  f"SpO2: {mean_spo2:.1f}% | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Time: {elapsed/60:.1f}m")
        
        # Checkpoint
        if episode % checkpoint_freq == 0:
            agent.save(str(checkpoint_path / f'checkpoint_{episode}'))
            logger.save()
        
        # Save best
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(str(checkpoint_path / 'best'))
    
    # Final save
    agent.save(str(checkpoint_path / 'final'))
    logger.save()
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    convert_to_tflite(
        agent.q_network,
        str(checkpoint_path / 'model.tflite'),
        quantize=True
    )
    
    print(f"\nTraining complete!")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best reward: {best_reward:.1f}")
    
    return agent


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=3000,
                       help='Max steps per episode')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--checkpoint-freq', type=int, default=100,
                       help='Checkpoint frequency')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    train_dqn(
        episodes=args.episodes,
        max_steps=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        checkpoint_freq=args.checkpoint_freq,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
