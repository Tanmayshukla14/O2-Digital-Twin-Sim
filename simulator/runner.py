"""
Simulation Runner
=================

Main entry point for running oxygen concentrator simulations.
Supports visualization, data logging, and validation.
"""

import argparse
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

from .environment import OxygenConcentratorEnv, EnvironmentConfig
from .physics.patient import PatientCondition


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    duration: float
    final_spo2: float
    mean_spo2: float
    min_spo2: float
    mean_purity: float
    min_purity: float
    total_energy_wh: float
    hypoxia_events: int
    passed_validation: bool
    validation_messages: List[str]


class SimulationRunner:
    """
    Runs oxygen concentrator simulations with optional control.
    
    Supports:
    - Open-loop simulation with fixed setpoints
    - Closed-loop with PID controller
    - AI agent inference (if model provided)
    - Data logging to files
    - Real-time visualization
    """
    
    def __init__(self, 
                 config: Optional[EnvironmentConfig] = None,
                 controller=None,
                 ai_agent=None):
        """
        Initialize simulation runner.
        
        Args:
            config: Environment configuration
            controller: Optional PID/Fuzzy controller
            ai_agent: Optional trained AI agent
        """
        self.env = OxygenConcentratorEnv(config)
        self.controller = controller
        self.ai_agent = ai_agent
        
        # Data logging
        self.history: List[Dict] = []
        
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset simulation."""
        self.env.reset(seed=seed)
        self.history = []
        
    def run(self, 
            duration: float = 300.0,
            visualize: bool = False,
            log_interval: float = 1.0,
            verbose: bool = True) -> SimulationResult:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration [s]
            visualize: Enable real-time console output
            log_interval: Data logging interval [s]
            verbose: Print progress messages
            
        Returns:
            SimulationResult with metrics
        """
        self.env.config.episode_duration = duration
        obs, info = self.env.reset()
        
        if verbose:
            print(f"Starting simulation: {duration}s, "
                  f"Patient: {info['condition']}")
        
        done = False
        step_count = 0
        last_log_time = 0.0
        
        # Metrics tracking
        spo2_values = []
        purity_values = []
        
        while not done:
            # Get action
            if self.ai_agent is not None:
                action = self.ai_agent.predict(obs)
            elif self.controller is not None:
                # Use controller to determine action
                action = self._controller_to_action(obs)
            else:
                # Default: maintain current setpoints
                action = 2
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            step_count += 1
            
            # Track metrics
            spo2_values.append(info['spo2'])
            purity_values.append(info['purity'])
            
            # Logging
            if self.env.elapsed_time - last_log_time >= log_interval:
                state = self.env.get_state_dict()
                state['reward'] = reward
                self.history.append(state)
                last_log_time = self.env.elapsed_time
                
                if visualize:
                    self.env.render()
        
        if verbose:
            print(f"Simulation complete: {step_count} steps, "
                  f"Final SpO2: {info['spo2']:.1f}%")
        
        # Compute results
        result = self._compute_results(spo2_values, purity_values)
        return result
    
    def _controller_to_action(self, obs: np.ndarray) -> int:
        """Convert controller output to discrete action."""
        if self.controller is None:
            return 2
        
        # Extract relevant state
        state = {
            'pressure': obs[0],
            'purity': obs[3],
            'spo2': obs[6],
            'spo2_trend': obs[7],
        }
        
        # Get controller output
        output = self.controller.compute(state, {
            'spo2_target': 94.0,
            'purity_target': 93.0,
        })
        
        # Map to discrete action
        pressure_adj = output.get('pressure_adjustment', 0)
        
        if pressure_adj < -0.1:
            return 0  # Large decrease
        elif pressure_adj < -0.02:
            return 1  # Small decrease
        elif pressure_adj > 0.1:
            return 4  # Large increase
        elif pressure_adj > 0.02:
            return 3  # Small increase
        else:
            return 2  # Maintain
    
    def _compute_results(self, 
                         spo2_values: List[float],
                         purity_values: List[float]) -> SimulationResult:
        """Compute simulation results and validation."""
        spo2_arr = np.array(spo2_values)
        purity_arr = np.array(purity_values)
        
        # Get final state
        final_state = self.env.get_state_dict()
        
        # Validation checks
        validation_messages = []
        passed = True
        
        # Check 1: Mean SpO2 >= 88%
        mean_spo2 = np.mean(spo2_arr)
        if mean_spo2 < 88:
            passed = False
            validation_messages.append(f"FAIL: Mean SpO2 {mean_spo2:.1f}% < 88%")
        else:
            validation_messages.append(f"PASS: Mean SpO2 {mean_spo2:.1f}% >= 88%")
        
        # Check 2: Min SpO2 >= 80% (no critical hypoxia)
        min_spo2 = np.min(spo2_arr)
        if min_spo2 < 80:
            passed = False
            validation_messages.append(f"FAIL: Min SpO2 {min_spo2:.1f}% < 80%")
        else:
            validation_messages.append(f"PASS: Min SpO2 {min_spo2:.1f}% >= 80%")
        
        # Check 3: Mean purity >= 87%
        mean_purity = np.mean(purity_arr)
        if mean_purity < 87:
            passed = False
            validation_messages.append(f"FAIL: Mean purity {mean_purity:.1f}% < 87%")
        else:
            validation_messages.append(f"PASS: Mean purity {mean_purity:.1f}% >= 87%")
        
        return SimulationResult(
            duration=self.env.elapsed_time,
            final_spo2=spo2_arr[-1] if len(spo2_arr) > 0 else 0,
            mean_spo2=mean_spo2,
            min_spo2=min_spo2,
            mean_purity=mean_purity,
            min_purity=np.min(purity_arr),
            total_energy_wh=final_state['compressor']['cumulative_energy_wh'],
            hypoxia_events=final_state['patient']['hypoxia_events'],
            passed_validation=passed,
            validation_messages=validation_messages
        )
    
    def save_history(self, filepath: str) -> None:
        """Save simulation history to file."""
        import json
        
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        history = convert(self.history)
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Saved history to {filepath}")


def main():
    """CLI entry point for simulation runner."""
    parser = argparse.ArgumentParser(
        description='Run oxygen concentrator simulation'
    )
    parser.add_argument('--duration', type=float, default=300,
                       help='Simulation duration in seconds')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable real-time visualization')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation checks')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for history data')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create and run simulation
    runner = SimulationRunner()
    
    if args.seed:
        np.random.seed(args.seed)
    
    result = runner.run(
        duration=args.duration,
        visualize=args.visualize,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"Duration: {result.duration:.1f}s")
    print(f"Final SpO2: {result.final_spo2:.1f}%")
    print(f"Mean SpO2: {result.mean_spo2:.1f}%")
    print(f"Min SpO2: {result.min_spo2:.1f}%")
    print(f"Mean Purity: {result.mean_purity:.1f}%")
    print(f"Total Energy: {result.total_energy_wh:.2f} Wh")
    print(f"Hypoxia Events: {result.hypoxia_events}")
    
    if args.validate:
        print("\nVALIDATION:")
        for msg in result.validation_messages:
            print(f"  {msg}")
        print(f"\nOverall: {'PASSED' if result.passed_validation else 'FAILED'}")
    
    # Save history if requested
    if args.output:
        runner.save_history(args.output)
    
    # Return exit code based on validation
    return 0 if result.passed_validation else 1


if __name__ == '__main__':
    exit(main())
