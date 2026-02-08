"""
PID Controller with Anti-Windup
===============================

Industrial-grade PID controller implementation with:
- Anti-windup (back-calculation and clamping)
- Derivative filtering
- Bumpless transfer
- Output rate limiting
- Auto-tuning interface

Designed for 1kHz real-time operation on embedded systems.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from abc import ABC, abstractmethod


@dataclass
class PIDConfig:
    """PID controller configuration."""
    
    # Gains
    kp: float = 1.0                   # Proportional gain
    ki: float = 0.1                   # Integral gain
    kd: float = 0.01                  # Derivative gain
    
    # Output limits
    output_min: float = 0.0           # Minimum output
    output_max: float = 1.0           # Maximum output
    
    # Anti-windup
    anti_windup_gain: float = 0.1     # Back-calculation gain (Kb)
    integral_limit: float = 10.0      # Integral term limit
    
    # Derivative filtering
    derivative_filter_tau: float = 0.01  # Derivative filter time constant [s]
    use_derivative_on_measurement: bool = True  # D on measurement vs error
    
    # Rate limiting
    output_rate_limit: float = 1.0    # Max output change per second
    
    # Setpoint weighting
    setpoint_weight_p: float = 1.0    # Proportional setpoint weight (b)
    setpoint_weight_d: float = 0.0    # Derivative setpoint weight (c)
    
    # Sample time
    dt: float = 0.001                 # Sample time [s] (1kHz default)


@dataclass
class PIDState:
    """PID controller internal state."""
    
    # Error terms
    error: float = 0.0
    error_integral: float = 0.0
    error_derivative: float = 0.0
    
    # Previous values
    prev_error: float = 0.0
    prev_measurement: float = 0.0
    prev_output: float = 0.0
    
    # Filtered derivative
    derivative_filtered: float = 0.0
    
    # Output components (for diagnostics)
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    output_raw: float = 0.0
    output_limited: float = 0.0
    
    # Status
    saturated: bool = False
    windup_active: bool = False


class PIDController:
    """
    Industrial PID controller with anti-windup.
    
    Features:
    - Back-calculation anti-windup prevents integrator saturation
    - First-order derivative filter reduces noise sensitivity
    - Derivative-on-measurement avoids setpoint kick
    - Output rate limiting for smooth actuator control
    - Setpoint weighting for tuning response characteristics
    
    Usage:
        pid = PIDController(PIDConfig(kp=2.0, ki=0.5, kd=0.1))
        output = pid.compute(setpoint=100, measurement=95)
    """
    
    def __init__(self, config: Optional[PIDConfig] = None):
        self.config = config or PIDConfig()
        self.state = PIDState()
        self._initialized = False
        
    def reset(self, initial_output: float = 0.0) -> None:
        """Reset controller state."""
        self.state = PIDState(prev_output=initial_output)
        self._initialized = False
    
    def compute(self, 
                setpoint: float, 
                measurement: float,
                dt: Optional[float] = None,
                feedforward: float = 0.0) -> float:
        """
        Compute PID control output.
        
        Args:
            setpoint: Desired value
            measurement: Current measured value
            dt: Sample time (uses config default if None)
            feedforward: Optional feedforward term
            
        Returns:
            Control output (limited to configured range)
        """
        cfg = self.config
        dt = dt or cfg.dt
        
        # Initialize on first call
        if not self._initialized:
            self.state.prev_measurement = measurement
            self.state.prev_error = setpoint - measurement
            self._initialized = True
        
        # === Error Calculation ===
        self.state.error = setpoint - measurement
        
        # === Proportional Term ===
        # With setpoint weighting: P = Kp * (b*setpoint - measurement)
        weighted_error_p = cfg.setpoint_weight_p * setpoint - measurement
        self.state.p_term = cfg.kp * weighted_error_p
        
        # === Derivative Term ===
        if cfg.use_derivative_on_measurement:
            # Derivative on measurement (avoids setpoint kick)
            derivative_input = -(measurement - self.state.prev_measurement) / dt
        else:
            # Derivative on error (with setpoint weighting)
            weighted_error_d = cfg.setpoint_weight_d * setpoint - measurement
            prev_weighted_d = cfg.setpoint_weight_d * setpoint - self.state.prev_measurement
            derivative_input = (weighted_error_d - prev_weighted_d) / dt
        
        # First-order derivative filter
        if cfg.derivative_filter_tau > 0:
            alpha = dt / (dt + cfg.derivative_filter_tau)
            self.state.derivative_filtered = (
                alpha * derivative_input + 
                (1 - alpha) * self.state.derivative_filtered
            )
        else:
            self.state.derivative_filtered = derivative_input
        
        self.state.d_term = cfg.kd * self.state.derivative_filtered
        
        # === Integral Term ===
        # Trapezoidal integration
        integral_increment = cfg.ki * (self.state.error + self.state.prev_error) * dt / 2
        
        # Add to integral (will be modified by anti-windup)
        new_integral = self.state.error_integral + integral_increment
        
        # Compute raw output for anti-windup calculation
        self.state.output_raw = (
            self.state.p_term + 
            cfg.ki * new_integral + 
            self.state.d_term + 
            feedforward
        )
        
        # === Output Limiting ===
        self.state.output_limited = np.clip(
            self.state.output_raw, 
            cfg.output_min, 
            cfg.output_max
        )
        
        self.state.saturated = (
            self.state.output_raw != self.state.output_limited
        )
        
        # === Anti-Windup (Back-Calculation) ===
        if self.state.saturated:
            # Back-calculation: reduce integral based on saturation
            saturation_error = self.state.output_limited - self.state.output_raw
            anti_windup_correction = cfg.anti_windup_gain * saturation_error * dt
            new_integral += anti_windup_correction / cfg.ki if cfg.ki != 0 else 0
            self.state.windup_active = True
        else:
            self.state.windup_active = False
        
        # Clamp integral
        self.state.error_integral = np.clip(
            new_integral, 
            -cfg.integral_limit, 
            cfg.integral_limit
        )
        self.state.i_term = cfg.ki * self.state.error_integral
        
        # === Rate Limiting ===
        if cfg.output_rate_limit > 0:
            max_change = cfg.output_rate_limit * dt
            output_change = self.state.output_limited - self.state.prev_output
            if abs(output_change) > max_change:
                self.state.output_limited = (
                    self.state.prev_output + 
                    np.sign(output_change) * max_change
                )
        
        # === Update State ===
        self.state.prev_error = self.state.error
        self.state.prev_measurement = measurement
        self.state.prev_output = self.state.output_limited
        
        return self.state.output_limited
    
    def set_gains(self, kp: float = None, ki: float = None, kd: float = None) -> None:
        """Update PID gains (bumpless transfer)."""
        if kp is not None:
            self.config.kp = kp
        if ki is not None:
            # Adjust integral to maintain output continuity
            if self.config.ki != 0 and ki != 0:
                self.state.error_integral *= self.config.ki / ki
            self.config.ki = ki
        if kd is not None:
            self.config.kd = kd
    
    def set_output_limits(self, min_output: float, max_output: float) -> None:
        """Update output limits."""
        self.config.output_min = min_output
        self.config.output_max = max_output
    
    def get_telemetry(self) -> dict:
        """Get controller telemetry for logging."""
        return {
            'error': self.state.error,
            'p_term': self.state.p_term,
            'i_term': self.state.i_term,
            'd_term': self.state.d_term,
            'output_raw': self.state.output_raw,
            'output': self.state.output_limited,
            'integral': self.state.error_integral,
            'saturated': self.state.saturated,
            'windup_active': self.state.windup_active,
        }


class CascadePIDController:
    """
    Cascade PID controller for pressure-flow control.
    
    Outer loop: SpO2/Purity control (slow)
    Inner loop: Pressure control (fast)
    """
    
    def __init__(self,
                 outer_config: Optional[PIDConfig] = None,
                 inner_config: Optional[PIDConfig] = None):
        
        # Outer loop: slower, controls setpoint
        self.outer = PIDController(outer_config or PIDConfig(
            kp=0.5,
            ki=0.01,
            kd=0.1,
            output_min=1.0,
            output_max=3.0,
            dt=0.1,  # 10 Hz
        ))
        
        # Inner loop: faster, controls actuator
        self.inner = PIDController(inner_config or PIDConfig(
            kp=2.0,
            ki=0.5,
            kd=0.05,
            output_min=0.0,
            output_max=1.0,
            dt=0.001,  # 1 kHz
        ))
    
    def reset(self) -> None:
        """Reset both controllers."""
        self.outer.reset()
        self.inner.reset()
    
    @property
    def outer_controller(self):
        """Alias for outer controller."""
        return self.outer
    
    @property
    def inner_controller(self):
        """Alias for inner controller."""
        return self.inner
    
    def compute(self, outer_setpoint: float, outer_measurement: float,
                inner_measurement: float, dt: float = 0.1) -> float:
        """Combined compute for dashboard compatibility."""
        pressure_setpoint = self.outer.compute(outer_setpoint, outer_measurement, dt=dt)
        return self.inner.compute(pressure_setpoint, inner_measurement, dt=dt*0.01)
    
    def compute_outer(self, 
                      setpoint: float, 
                      measurement: float) -> float:
        """
        Compute outer loop (slow - call at 10Hz).
        
        Args:
            setpoint: Target SpO2 or purity
            measurement: Current SpO2 or purity
            
        Returns:
            Pressure setpoint for inner loop
        """
        return self.outer.compute(setpoint, measurement)
    
    def compute_inner(self,
                      pressure_setpoint: float,
                      pressure_measurement: float) -> float:
        """
        Compute inner loop (fast - call at 1kHz).
        
        Args:
            pressure_setpoint: Target pressure from outer loop
            pressure_measurement: Current pressure
            
        Returns:
            PWM duty cycle for motor
        """
        return self.inner.compute(pressure_setpoint, pressure_measurement)
    
    def get_telemetry(self) -> dict:
        """Get telemetry from both loops."""
        return {
            'outer': self.outer.get_telemetry(),
            'inner': self.inner.get_telemetry(),
        }
