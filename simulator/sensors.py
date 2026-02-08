"""
Sensor Simulation Module
========================

Models realistic sensor behavior including:
- Measurement noise (Gaussian, uniform)
- Sensor drift over time
- Measurement delay/latency
- Saturation and quantization
- Fault injection for testing
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from collections import deque
from enum import Enum


class SensorType(Enum):
    """Types of sensors in the system."""
    PRESSURE = 0
    OXYGEN = 1
    TEMPERATURE = 2
    FLOW = 3
    HUMIDITY = 4
    CURRENT = 5
    VOLTAGE = 6


class SensorFault(Enum):
    """Sensor fault modes for testing."""
    NONE = 0
    STUCK = 1           # Sensor stuck at last value
    BIAS = 2            # Constant offset
    DRIFT = 3           # Time-varying drift
    NOISE_HIGH = 4      # Excessive noise
    INTERMITTENT = 5    # Random dropouts
    SATURATED_HIGH = 6  # Stuck at max
    SATURATED_LOW = 7   # Stuck at min


@dataclass
class SensorConfig:
    """Configuration for a single sensor."""
    
    sensor_type: SensorType = SensorType.PRESSURE
    name: str = "sensor"
    
    # Range
    min_value: float = 0.0
    max_value: float = 100.0
    
    # Accuracy
    accuracy_percent: float = 1.0     # % of full scale
    resolution: float = 0.01          # Minimum detectable change
    
    # Noise characteristics
    noise_std: float = 0.1            # Gaussian noise std dev
    noise_uniform_range: float = 0.0  # Uniform noise range (±)
    
    # Dynamics
    delay_samples: int = 1            # Measurement delay in samples
    time_constant: float = 0.01       # First-order lag time constant [s]
    
    # Drift
    drift_rate: float = 0.0           # Drift per hour [units/hr]
    drift_random_walk: float = 0.0    # Random walk coefficient
    
    # Quantization
    adc_bits: int = 12                # ADC resolution
    
    # Fault injection
    fault_mode: SensorFault = SensorFault.NONE
    fault_magnitude: float = 0.0


@dataclass
class SensorState:
    """Dynamic state of a sensor."""
    
    true_value: float = 0.0           # Actual physical value
    measured_value: float = 0.0       # Sensor output
    filtered_value: float = 0.0       # After filtering
    
    drift_accumulated: float = 0.0    # Accumulated drift
    is_valid: bool = True             # Sensor health flag
    fault_active: bool = False        # Fault condition active
    
    sample_count: int = 0             # Total samples taken


class Sensor:
    """
    Realistic sensor model with noise, delay, and fault injection.
    
    Used to simulate the imperfect measurements that the control
    system must handle in real deployment.
    """
    
    def __init__(self, config: SensorConfig):
        self.config = config
        self.state = SensorState()
        
        # Delay buffer
        self._delay_buffer = deque(
            [0.0] * (config.delay_samples + 1), 
            maxlen=config.delay_samples + 1
        )
        
        # Random state for reproducibility
        self._rng = np.random.default_rng()
        
        # Precompute quantization levels
        self._quant_levels = 2 ** config.adc_bits
        self._quant_step = (config.max_value - config.min_value) / self._quant_levels
        
    def reset(self, seed: Optional[int] = None) -> SensorState:
        """Reset sensor to initial state."""
        self.state = SensorState()
        self._delay_buffer = deque(
            [0.0] * (self.config.delay_samples + 1),
            maxlen=self.config.delay_samples + 1
        )
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self.state
    
    def measure(self, true_value: float, dt: float) -> float:
        """
        Generate sensor measurement from true value.
        
        Args:
            true_value: Actual physical value
            dt: Time step [s]
            
        Returns:
            Measured (noisy, delayed) sensor value
        """
        self.state.true_value = true_value
        self.state.sample_count += 1
        
        cfg = self.config
        
        # --- Apply Sensor Dynamics (first-order lag) ---
        if cfg.time_constant > 0:
            alpha = dt / (dt + cfg.time_constant)
            self.state.filtered_value = (
                alpha * true_value + 
                (1 - alpha) * self.state.filtered_value
            )
        else:
            self.state.filtered_value = true_value
        
        value = self.state.filtered_value
        
        # --- Add Drift ---
        if cfg.drift_rate != 0:
            self.state.drift_accumulated += cfg.drift_rate * dt / 3600
        
        if cfg.drift_random_walk != 0:
            self.state.drift_accumulated += (
                self._rng.normal(0, cfg.drift_random_walk * np.sqrt(dt))
            )
        
        value += self.state.drift_accumulated
        
        # --- Add Noise ---
        if cfg.noise_std > 0:
            value += self._rng.normal(0, cfg.noise_std)
        
        if cfg.noise_uniform_range > 0:
            value += self._rng.uniform(-cfg.noise_uniform_range, cfg.noise_uniform_range)
        
        # --- Apply Faults ---
        value = self._apply_fault(value)
        
        # --- Quantization ---
        value = self._quantize(value)
        
        # --- Saturation ---
        value = np.clip(value, cfg.min_value, cfg.max_value)
        
        # --- Apply Delay ---
        self._delay_buffer.append(value)
        delayed_value = self._delay_buffer[0]
        
        self.state.measured_value = delayed_value
        
        return delayed_value
    
    def _apply_fault(self, value: float) -> float:
        """Apply fault mode to measurement."""
        fault = self.config.fault_mode
        
        if fault == SensorFault.NONE:
            self.state.fault_active = False
            return value
        
        self.state.fault_active = True
        
        if fault == SensorFault.STUCK:
            return self.state.measured_value  # Return last value
        
        elif fault == SensorFault.BIAS:
            return value + self.config.fault_magnitude
        
        elif fault == SensorFault.DRIFT:
            # Accelerated drift
            drift = self.config.fault_magnitude * self.state.sample_count / 1000
            return value + drift
        
        elif fault == SensorFault.NOISE_HIGH:
            noise = self._rng.normal(0, self.config.fault_magnitude)
            return value + noise
        
        elif fault == SensorFault.INTERMITTENT:
            if self._rng.random() < 0.1:  # 10% dropout
                return float('nan')
            return value
        
        elif fault == SensorFault.SATURATED_HIGH:
            return self.config.max_value
        
        elif fault == SensorFault.SATURATED_LOW:
            return self.config.min_value
        
        return value
    
    def _quantize(self, value: float) -> float:
        """Apply ADC quantization."""
        normalized = (value - self.config.min_value) / (
            self.config.max_value - self.config.min_value)
        quantized = round(normalized * self._quant_levels) / self._quant_levels
        return self.config.min_value + quantized * (
            self.config.max_value - self.config.min_value)
    
    def set_fault(self, fault: SensorFault, magnitude: float = 0.0) -> None:
        """Inject sensor fault."""
        self.config.fault_mode = fault
        self.config.fault_magnitude = magnitude
    
    def clear_fault(self) -> None:
        """Clear active fault."""
        self.config.fault_mode = SensorFault.NONE
        self.state.fault_active = False
    
    def is_valid(self) -> bool:
        """Check if sensor reading is valid."""
        if self.state.fault_active:
            return False
        if np.isnan(self.state.measured_value):
            return False
        return True


class SensorArray:
    """
    Collection of sensors for the oxygen concentrator system.
    
    Provides a unified interface for all system sensors with
    realistic noise, dynamics, and fault injection.
    """
    
    def __init__(self):
        self.sensors: Dict[str, Sensor] = {}
        self._setup_default_sensors()
    
    def _setup_default_sensors(self) -> None:
        """Create default sensor suite for oxygen concentrator."""
        
        # Tank pressure sensor
        self.sensors['pressure_tank'] = Sensor(SensorConfig(
            sensor_type=SensorType.PRESSURE,
            name='pressure_tank',
            min_value=0.0,
            max_value=4.0,  # bar
            accuracy_percent=0.5,
            noise_std=0.01,
            delay_samples=2,
            time_constant=0.005,
            adc_bits=12
        ))
        
        # PSA bed pressure sensors
        self.sensors['pressure_bed_a'] = Sensor(SensorConfig(
            sensor_type=SensorType.PRESSURE,
            name='pressure_bed_a',
            min_value=0.0,
            max_value=4.0,
            accuracy_percent=0.5,
            noise_std=0.02,
            delay_samples=2,
            time_constant=0.01,
            adc_bits=12
        ))
        
        self.sensors['pressure_bed_b'] = Sensor(SensorConfig(
            sensor_type=SensorType.PRESSURE,
            name='pressure_bed_b',
            min_value=0.0,
            max_value=4.0,
            accuracy_percent=0.5,
            noise_std=0.02,
            delay_samples=2,
            time_constant=0.01,
            adc_bits=12
        ))
        
        # Oxygen concentration sensor (zirconia cell)
        self.sensors['oxygen'] = Sensor(SensorConfig(
            sensor_type=SensorType.OXYGEN,
            name='oxygen',
            min_value=0.0,
            max_value=100.0,  # %
            accuracy_percent=1.0,
            noise_std=0.5,
            delay_samples=10,  # Slower response
            time_constant=0.5,  # 500ms time constant
            drift_rate=0.1,  # Slow drift
            adc_bits=10
        ))
        
        # Flow sensor
        self.sensors['flow'] = Sensor(SensorConfig(
            sensor_type=SensorType.FLOW,
            name='flow',
            min_value=0.0,
            max_value=15.0,  # LPM
            accuracy_percent=2.0,
            noise_std=0.1,
            delay_samples=3,
            time_constant=0.02,
            adc_bits=12
        ))
        
        # Temperature sensors
        self.sensors['temp_motor'] = Sensor(SensorConfig(
            sensor_type=SensorType.TEMPERATURE,
            name='temp_motor',
            min_value=-20.0,
            max_value=120.0,  # °C
            accuracy_percent=0.5,
            noise_std=0.3,
            delay_samples=5,
            time_constant=1.0,  # Thermal lag
            adc_bits=12
        ))
        
        self.sensors['temp_ambient'] = Sensor(SensorConfig(
            sensor_type=SensorType.TEMPERATURE,
            name='temp_ambient',
            min_value=-20.0,
            max_value=60.0,
            accuracy_percent=0.5,
            noise_std=0.2,
            delay_samples=5,
            time_constant=2.0,
            adc_bits=12
        ))
        
        # Humidity sensor
        self.sensors['humidity'] = Sensor(SensorConfig(
            sensor_type=SensorType.HUMIDITY,
            name='humidity',
            min_value=0.0,
            max_value=100.0,  # %RH
            accuracy_percent=3.0,
            noise_std=1.0,
            delay_samples=10,
            time_constant=5.0,
            adc_bits=10
        ))
        
        # Motor current sensor
        self.sensors['motor_current'] = Sensor(SensorConfig(
            sensor_type=SensorType.CURRENT,
            name='motor_current',
            min_value=0.0,
            max_value=25.0,  # A
            accuracy_percent=1.0,
            noise_std=0.05,
            delay_samples=1,
            time_constant=0.001,
            adc_bits=12
        ))
    
    def measure_all(self, true_values: Dict[str, float], dt: float) -> Dict[str, float]:
        """
        Get measurements from all sensors.
        
        Args:
            true_values: Dict of true physical values keyed by sensor name
            dt: Time step [s]
            
        Returns:
            Dict of measured (noisy) values
        """
        measurements = {}
        for name, sensor in self.sensors.items():
            if name in true_values:
                measurements[name] = sensor.measure(true_values[name], dt)
            else:
                measurements[name] = sensor.state.measured_value
        return measurements
    
    def reset_all(self, seed: Optional[int] = None) -> None:
        """Reset all sensors."""
        for i, sensor in enumerate(self.sensors.values()):
            sensor.reset(seed + i if seed else None)
    
    def inject_fault(self, sensor_name: str, fault: SensorFault, 
                    magnitude: float = 0.0) -> None:
        """Inject fault into specific sensor."""
        if sensor_name in self.sensors:
            self.sensors[sensor_name].set_fault(fault, magnitude)
    
    def clear_all_faults(self) -> None:
        """Clear all sensor faults."""
        for sensor in self.sensors.values():
            sensor.clear_fault()
    
    def get_validity(self) -> Dict[str, bool]:
        """Get validity status of all sensors."""
        return {name: sensor.is_valid() for name, sensor in self.sensors.items()}
    
    def get_state_dict(self) -> dict:
        """Return all sensor measurements as dictionary."""
        return {
            name: {
                'measured': sensor.state.measured_value,
                'true': sensor.state.true_value,
                'valid': sensor.is_valid(),
                'fault': sensor.state.fault_active
            }
            for name, sensor in self.sensors.items()
        }
