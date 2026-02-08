"""
Gym-Compatible Reinforcement Learning Environment
=================================================

Provides a Gymnasium-compatible environment for training
DQN/PPO agents to optimize oxygen concentrator control.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from .physics.compressor import DiaphragmCompressor, CompressorConfig
from .physics.psa import PSASystem, PSAConfig
from .physics.patient import PatientModel, PatientConfig, PatientCondition, ActivityLevel
from .sensors import SensorArray


@dataclass
class EnvironmentConfig:
    """Configuration for RL environment."""
    
    # Simulation parameters
    dt: float = 0.001                     # Physics timestep [s]
    control_dt: float = 0.1               # Control loop period [s] (10 Hz AI)
    episode_duration: float = 300.0       # Episode length [s]
    
    # Initial conditions
    initial_tank_pressure: float = 1.5e5  # [Pa]
    initial_purity: float = 0.93          # [fraction]
    initial_spo2: float = 92.0            # [%]
    
    # Randomization (for robust training)
    randomize_patient: bool = True
    randomize_environment: bool = True
    
    # Reward weights
    reward_spo2_weight: float = 10.0
    reward_purity_weight: float = 5.0 
    reward_energy_weight: float = 0.1
    reward_stability_weight: float = 1.0
    
    # Termination conditions
    terminate_on_critical_spo2: bool = True
    critical_spo2_threshold: float = 80.0


class OxygenConcentratorEnv(gym.Env):
    """
    Gym-compatible environment for oxygen concentrator control.
    
    Observation Space (14 dimensions):
        0: Tank pressure [bar] (0.5-3.0)
        1: PSA bed A pressure [bar]
        2: PSA bed B pressure [bar]
        3: Oxygen purity [%] (21-99)
        4: Purity rate of change [%/s]
        5: Product flow [LPM]
        6: Patient SpO2 [%]
        7: SpO2 trend [%/min]
        8: Motor temperature [°C]
        9: Compressor health [0-1]
        10: Zeolite efficiency [0-1]
        11: Instantaneous power [W]
        12: PSA phase [0-3]
        13: PSA cycle time [s]
    
    Action Space (Discrete, 9 actions):
        0: Decrease pressure setpoint (large)
        1: Decrease pressure setpoint (small)
        2: Maintain current setpoint
        3: Increase pressure setpoint (small)
        4: Increase pressure setpoint (large)
        5: Decrease PSA cycle time
        6: Increase PSA cycle time
        7: Decrease flow rate
        8: Increase flow rate
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        super().__init__()
        
        self.config = config or EnvironmentConfig()
        
        # Initialize physics models
        self.compressor = DiaphragmCompressor()
        self.psa = PSASystem()
        self.patient = PatientModel()
        self.sensors = SensorArray()
        
        # Control state
        self.pressure_setpoint = 2.0e5  # Pa
        self.flow_setpoint = 3.0        # LPM
        self.pwm_duty = 0.5             # Motor duty cycle
        
        # Simulation counters
        self.current_step = 0
        self.elapsed_time = 0.0
        self.physics_steps_per_control = int(self.config.control_dt / self.config.dt)
        
        # === Define Spaces ===
        # Observation: normalized to roughly [-1, 1] range
        self.observation_space = spaces.Box(
            low=np.array([0.5, 0.5, 0.5, 21, -10, 0, 50, -5, 0, 0, 0.5, 0, 0, 10], dtype=np.float32),
            high=np.array([3.0, 3.0, 3.0, 99, 10, 15, 100, 5, 100, 1, 1, 500, 3, 60], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: discrete
        self.action_space = spaces.Discrete(9)
        
        # Action mappings
        self._action_effects = {
            0: ('pressure', -0.2e5),     # Large decrease
            1: ('pressure', -0.05e5),    # Small decrease
            2: ('pressure', 0.0),        # Maintain
            3: ('pressure', 0.05e5),     # Small increase
            4: ('pressure', 0.2e5),      # Large increase
            5: ('cycle_time', -2.0),     # Decrease cycle
            6: ('cycle_time', 2.0),      # Increase cycle
            7: ('flow', -0.5),           # Decrease flow
            8: ('flow', 0.5),            # Increase flow
        }
        
        # Episode tracking
        self._episode_reward = 0.0
        self._episode_info = {}
        
    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset physics models
        self.compressor.reset()
        self.psa.reset()
        self.sensors.reset_all(seed)
        
        # Randomize patient condition for robust training
        if self.config.randomize_patient:
            conditions = [PatientCondition.COPD_MILD, 
                         PatientCondition.COPD_MODERATE,
                         PatientCondition.COPD_SEVERE,
                         PatientCondition.PNEUMONIA]
            condition = self.np_random.choice(conditions)
            self.patient.reset(condition=condition)
        else:
            self.patient.reset()
        
        # Randomize environment
        if self.config.randomize_environment:
            self.psa.state.ambient_humidity = self.np_random.uniform(30, 70)
        
        # Reset control state
        self.pressure_setpoint = 2.0e5
        self.flow_setpoint = 3.0
        self.pwm_duty = 0.5
        
        # Reset counters
        self.current_step = 0
        self.elapsed_time = 0.0
        self._episode_reward = 0.0
        
        # Get initial observation
        obs = self._get_observation()
        info = {'condition': self.patient.config.condition.name}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one control step (runs multiple physics steps).
        
        Args:
            action: Discrete action index
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Apply action
        self._apply_action(action)
        
        # Run physics simulation for control period
        for _ in range(self.physics_steps_per_control):
            self._physics_step()
            self.elapsed_time += self.config.dt
        
        self.current_step += 1
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._compute_reward()
        self._episode_reward += reward
        
        # Check termination
        terminated = False
        truncated = False
        
        # Critical SpO2 termination
        if (self.config.terminate_on_critical_spo2 and 
            self.patient.state.spo2 < self.config.critical_spo2_threshold):
            terminated = True
        
        # Episode timeout
        if self.elapsed_time >= self.config.episode_duration:
            truncated = True
        
        info = {
            'elapsed_time': self.elapsed_time,
            'spo2': self.patient.state.spo2,
            'purity': self.psa.state.o2_purity * 100,
            'pressure': self.compressor.state.outlet_pressure / 1e5,
            'power': self.compressor.state.instantaneous_power,
            'episode_reward': self._episode_reward,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action: int) -> None:
        """Apply action to control setpoints."""
        effect_type, delta = self._action_effects[action]
        
        if effect_type == 'pressure':
            self.pressure_setpoint += delta
            self.pressure_setpoint = np.clip(self.pressure_setpoint, 1.0e5, 3.0e5)
            
        elif effect_type == 'cycle_time':
            self.psa.set_cycle_time(self.psa.config.cycle_time + delta)
            
        elif effect_type == 'flow':
            self.flow_setpoint += delta
            self.flow_setpoint = np.clip(self.flow_setpoint, 0.5, 10.0)
    
    def _physics_step(self) -> None:
        """Execute one physics timestep."""
        dt = self.config.dt
        
        # Simple proportional control for motor PWM
        pressure_error = self.pressure_setpoint - self.compressor.state.outlet_pressure
        self.pwm_duty += 0.001 * pressure_error / 1e5
        self.pwm_duty = np.clip(self.pwm_duty, 0.0, 1.0)
        
        # Update compressor
        self.compressor.step(
            dt=dt,
            pwm_duty=self.pwm_duty,
            back_pressure=self.psa.state.bed_a_pressure
        )
        
        # Update PSA (using simplified step with correct parameters)
        self.psa.step(
            inlet_pressure=self.compressor.state.outlet_pressure / 1e5,  # Convert Pa to bar
            inlet_n2_fraction=0.79,
            inlet_o2_fraction=0.21,
            dt=dt
        )
        
        # Update patient (using simplified step with correct parameters)
        self.patient.step(
            delivered_o2_lpm=self.flow_setpoint,
            o2_purity=self.psa.state.o2_purity,
            dt=dt
        )
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector with sensor noise."""
        # Get true values
        true_values = {
            'pressure_tank': self.compressor.state.outlet_pressure / 1e5,
            'pressure_bed_a': self.psa.state.bed_a_pressure / 1e5,
            'pressure_bed_b': self.psa.state.bed_b_pressure / 1e5,
            'oxygen': self.psa.state.o2_purity * 100,
            'flow': self.flow_setpoint,
            'temp_motor': self.compressor.state.motor_temperature,
        }
        
        # Apply sensor noise
        measurements = self.sensors.measure_all(true_values, self.config.control_dt)
        
        obs = np.array([
            measurements.get('pressure_tank', true_values['pressure_tank']),
            measurements.get('pressure_bed_a', true_values['pressure_bed_a']),
            measurements.get('pressure_bed_b', true_values['pressure_bed_b']),
            measurements.get('oxygen', true_values['oxygen']),
            self.psa.state.o2_purity_rate * 100,
            measurements.get('flow', self.flow_setpoint),
            self.patient.state.spo2,
            self.patient.state.spo2_trend,
            measurements.get('temp_motor', true_values['temp_motor']),
            self.compressor.state.health_index,
            self.psa.state.zeolite_efficiency,
            self.compressor.state.instantaneous_power,
            float(self.psa.state.current_phase.value),
            self.psa.config.cycle_time,
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        cfg = self.config
        
        spo2 = self.patient.state.spo2
        purity = self.psa.state.o2_purity * 100
        power = self.compressor.state.instantaneous_power
        purity_rate = abs(self.psa.state.o2_purity_rate)
        
        # SpO2 reward (critical)
        if spo2 < 88:
            r_spo2 = -100.0 * (88 - spo2)
        elif spo2 < 92:
            r_spo2 = -10.0 * (92 - spo2)
        elif spo2 <= 96:
            r_spo2 = 10.0  # Target range
        else:
            r_spo2 = -5.0 * (spo2 - 96)  # Avoid over-oxygenation
        
        # Purity reward
        if purity < 87:
            r_purity = -50.0 * (87 - purity)
        elif purity >= 90:
            r_purity = 5.0
        else:
            r_purity = 0.0
        
        # Energy efficiency (lower is better)
        r_energy = -0.1 * power / 100
        
        # Stability (penalize oscillations)
        r_stability = -1.0 * purity_rate * 10
        
        total = (
            cfg.reward_spo2_weight * r_spo2 / 10 +
            cfg.reward_purity_weight * r_purity / 5 +
            cfg.reward_energy_weight * r_energy +
            cfg.reward_stability_weight * r_stability
        )
        
        return float(total)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get full system state for logging."""
        return {
            'time': self.elapsed_time,
            'compressor': self.compressor.get_state_dict(),
            'psa': self.psa.get_state_dict(),
            'patient': self.patient.get_state_dict(),
            'control': {
                'pressure_setpoint': self.pressure_setpoint / 1e5,
                'flow_setpoint': self.flow_setpoint,
                'pwm_duty': self.pwm_duty,
            }
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment (placeholder)."""
        if mode == 'human':
            print(f"t={self.elapsed_time:.1f}s | "
                  f"SpO2={self.patient.state.spo2:.1f}% | "
                  f"O2={self.psa.state.o2_purity*100:.1f}% | "
                  f"P={self.compressor.state.outlet_pressure/1e5:.2f}bar | "
                  f"Power={self.compressor.state.instantaneous_power:.1f}W")
        return None
