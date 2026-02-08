"""
Diaphragm Compressor Physics Model
==================================

Models the thermodynamic behavior of an oil-free diaphragm compressor
including pressure dynamics, thermal effects, and mechanical degradation.

Mathematical Basis:
- Polytropic compression: P1*V1^n = P2*V2^n
- Diaphragm displacement: x(t) = (S/2)(1 - cos(ωt))
- Thermal dynamics: dT/dt = (1/Cth)(Ploss - (T-Tamb)/Rth)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class CompressorConfig:
    """Configuration parameters for diaphragm compressor."""
    
    # Geometric parameters (Industrial / Medical Grade)
    stroke_length: float = 0.018          # Diaphragm stroke [m] - increased for higher compression
    diaphragm_area: float = 0.003         # Effective diaphragm area [m²] - larger displacement
    dead_volume: float = 3e-5             # Dead volume [m³] - tighter tolerances
    
    # Thermodynamic parameters
    polytropic_n: float = 1.3             # Polytropic index (1.0=isothermal, 1.4=adiabatic)
    gamma: float = 1.4                    # Specific heat ratio
    R_gas: float = 287.0                  # Gas constant for air [J/(kg·K)]
    
    # --- Motor Specifications ---
    # Using a high-torque BLDC motor to handle the 3.0 bar startup load.
    # Standard motors stall at ~2 bar, so we need extra torque (Kt=0.5).
    motor_resistance: float = 1.2         # Winding resistance [Ω]
    motor_inductance: float = 0.005       # Inductance [H]
    motor_kt: float = 0.5                 # Torque constant [N·m/A] - Industrial grade
    motor_ke: float = 0.5                 # Back-EMF [V/(rad/s)]
    motor_inertia: float = 0.002          # Rotor inertia [kg·m²]
    motor_friction: float = 0.002         # Bearings/Fan friction
    
    # --- Thermal Constraints ---
    thermal_mass: float = 500.0           # Aluminum block mass [J/K]
    thermal_resistance: float = 2.5       # Cooling efficiency [K/W]
    ambient_temperature: float = 25.0     # Default room temp [°C]
    max_temperature: float = 95.0         # Maximum safe temperature [°C]
    
    # Operating limits
    nominal_rpm: float = 2000.0           # Nominal speed [RPM] - faster operation

    max_rpm: float = 2500.0               # Maximum speed [RPM]
    min_pressure: float = 0.5e5           # Minimum outlet pressure [Pa]
    max_pressure: float = 3.0e5           # Maximum outlet pressure [Pa]
    
    # Degradation parameters
    wear_coefficient: float = 1e-12       # Archard wear coefficient
    initial_health: float = 1.0           # Initial health index


@dataclass
class CompressorState:
    """Dynamic state of the compressor."""
    
    # Mechanical state
    crank_angle: float = 0.0              # Crank angle [rad]
    angular_velocity: float = 0.0         # Angular velocity [rad/s]
    
    # Pressure state
    cylinder_pressure: float = 1.01325e5  # Cylinder pressure [Pa]
    outlet_pressure: float = 1.01325e5    # Outlet manifold pressure [Pa]
    
    # Thermal state
    motor_temperature: float = 25.0       # Motor temperature [°C]
    
    # Electrical state
    motor_current: float = 0.0            # Motor current [A]
    
    # Flow state
    mass_flow_rate: float = 0.0           # Mass flow rate [kg/s]
    volumetric_flow: float = 0.0          # Volumetric flow [m³/s]
    
    # Health state
    health_index: float = 1.0             # Health [0-1]
    total_cycles: int = 0                 # Total compression cycles
    
    # Energy
    instantaneous_power: float = 0.0      # Electrical power [W]
    cumulative_energy: float = 0.0        # Cumulative energy [J]


class DiaphragmCompressor:
    """
    Physics-based model of an oil-free diaphragm compressor.
    
    The compressor uses a motor-driven eccentric mechanism to oscillate
    a flexible diaphragm, creating pressure pulses for compression.
    
    Key Features:
    - Polytropic compression modeling
    - Thermal dynamics with heat dissipation
    - Mechanical wear and degradation
    - Valve dynamics (inlet/outlet)
    - Motor electrical dynamics
    """
    
    def __init__(self, config: Optional[CompressorConfig] = None):
        self.config = config or CompressorConfig()
        self.state = CompressorState(
            motor_temperature=self.config.ambient_temperature,
            health_index=self.config.initial_health
        )
        
        # Precompute derived parameters
        self._swept_volume = self.config.stroke_length * self.config.diaphragm_area
        self._total_volume = self.config.dead_volume + self._swept_volume
        self._omega_nominal = self.config.nominal_rpm * 2 * np.pi / 60
        
        # Valve state (simple on/off model)
        self._inlet_valve_open = False
        self._outlet_valve_open = False
        
        # Previous cycle tracking for flow calculation
        self._prev_angle = 0.0
        self._cycle_mass = 0.0
        
    def reset(self, 
              initial_temperature: float = None,
              initial_health: float = None) -> CompressorState:
        """Reset compressor to initial conditions."""
        self.state = CompressorState(
            motor_temperature=initial_temperature or self.config.ambient_temperature,
            health_index=initial_health or self.config.initial_health
        )
        self._prev_angle = 0.0
        self._cycle_mass = 0.0
        return self.state
    
    def step(self, 
             dt: float, 
             pwm_duty: float, 
             back_pressure: float,
             inlet_pressure: float = 1.01325e5) -> CompressorState:
        """
        Advance simulation by one timestep.
        
        Args:
            dt: Time step [s]
            pwm_duty: Motor PWM duty cycle [0-1]
            back_pressure: Downstream pressure (tank) [Pa]
            inlet_pressure: Inlet (atmospheric) pressure [Pa]
            
        Returns:
            Updated compressor state
        """
        # Input validation and clamping
        if dt <= 0:
            # For a physics step, dt must be positive. Returning current state or raising error.
            # Returning current state to avoid breaking simulation if dt becomes zero/negative.
            return self.state
            
        pwm_duty = float(np.clip(pwm_duty, 0.0, 1.0))
        if np.isnan(pwm_duty):
            pwm_duty = 0.0
            
        if back_pressure < 0:
            back_pressure = 101325.0  # Default to 1 atm if invalid
        if np.isnan(back_pressure) or np.isinf(back_pressure):
            back_pressure = 101325.0
        
        if inlet_pressure < 0:
            inlet_pressure = 101325.0 # Default to 1 atm if invalid
        if np.isnan(inlet_pressure) or np.isinf(inlet_pressure):
            inlet_pressure = 101325.0
            
        cfg = self.config
        
        # --- Motor Dynamics ---
        supply_voltage = pwm_duty * 24.0  # Assume 24V supply
        back_emf = cfg.motor_ke * self.state.angular_velocity
        
        # Motor current (simplified: ignore inductance for faster dynamics)
        self.state.motor_current = (supply_voltage - back_emf) / cfg.motor_resistance
        self.state.motor_current = np.clip(self.state.motor_current, 0, 20)  # Current limit
        
        # Motor torque
        motor_torque = cfg.motor_kt * self.state.motor_current
        
        # Load torque from compression
        compression_torque = self._compute_compression_torque(inlet_pressure)
        
        # Angular acceleration
        friction_torque = cfg.motor_friction * self.state.angular_velocity
        net_torque = motor_torque - compression_torque - friction_torque
        angular_accel = net_torque / cfg.motor_inertia
        
        # Integrate angular velocity and position
        self.state.angular_velocity += angular_accel * dt
        self.state.angular_velocity = np.clip(
            self.state.angular_velocity, 
            0, 
            cfg.max_rpm * 2 * np.pi / 60
        )
        
        self.state.crank_angle += self.state.angular_velocity * dt
        
        # Track cycles
        if self.state.crank_angle >= 2 * np.pi:
            self.state.crank_angle -= 2 * np.pi
            self.state.total_cycles += 1
            self._cycle_mass = 0.0
        
        # --- Compression Dynamics ---
        displacement = self._compute_displacement()
        current_volume = self._total_volume - self.config.diaphragm_area * displacement
        
        # Polytropic pressure
        if current_volume > 0:
            compression_ratio = self._total_volume / current_volume
            self.state.cylinder_pressure = inlet_pressure * (compression_ratio ** cfg.polytropic_n)
        
        # --- Valve Dynamics ---
        # Inlet valve: opens when cylinder pressure < inlet pressure (suction)
        self._inlet_valve_open = self.state.cylinder_pressure < inlet_pressure * 0.98
        
        # Outlet valve: opens when cylinder pressure > back pressure (discharge)
        self._outlet_valve_open = self.state.cylinder_pressure > back_pressure * 1.02
        
        # --- Mass Flow Calculation ---
        if self._outlet_valve_open:
            # Discharge flow (simplified orifice equation)
            delta_p = self.state.cylinder_pressure - back_pressure
            if delta_p > 0:
                # Cv = 0.025 (medical-grade valve with higher flow capacity)
                rho = self.state.cylinder_pressure / (cfg.R_gas * (cfg.ambient_temperature + 273.15))
                self.state.mass_flow_rate = 0.025 * np.sqrt(2 * rho * delta_p)
                self._cycle_mass += self.state.mass_flow_rate * dt
        else:
            self.state.mass_flow_rate = 0.0
        
        # Convert to volumetric flow at standard conditions
        rho_std = 1.01325e5 / (cfg.R_gas * 293.15)  # Standard density
        self.state.volumetric_flow = self.state.mass_flow_rate / rho_std
        
        # Update outlet pressure (realistic tank model)
        # Tank accumulates mass from compressor, loses mass to PSA consumption
        tank_volume = 0.010  # 10 liter tank [m³] - larger buffer for stable pressure
        temp_kelvin = self.config.ambient_temperature + 273.15
        
        # Current tank mass from ideal gas law: m = PV/(RT)
        tank_mass = back_pressure * tank_volume / (cfg.R_gas * temp_kelvin)
        
        # Add mass from compressor discharge
        if self.state.mass_flow_rate > 0:
            tank_mass += self.state.mass_flow_rate * dt
        
        # Subtract mass consumed by PSA
        # For 10-15 LPM O2 output: ~0.0003 kg/s air consumption (industrial capacity)
        consumption_rate = 0.0003  # kg/s - high capacity system
        tank_mass -= consumption_rate * dt
        tank_mass = max(tank_mass, 0.0001)  # Prevent negative mass
        
        # New pressure from ideal gas law: P = mRT/V
        self.state.outlet_pressure = tank_mass * cfg.R_gas * temp_kelvin / tank_volume
        self.state.outlet_pressure = np.clip(self.state.outlet_pressure, 1.01325e5, cfg.max_pressure)
        
        # --- Thermal Dynamics ---
        # Power loss = I²R + mechanical friction
        power_loss = (self.state.motor_current ** 2) * cfg.motor_resistance
        power_loss += cfg.motor_friction * (self.state.angular_velocity ** 2)
        
        # Thermal dynamics (First-order lumped capacitance model)
        # We model the motor as a single thermal mass (aluminum block).
        # Heat is generated by I²R losses and friction.
        # Heat is lost to ambient via convection (thermal resistance).
        dT_dt = (1 / cfg.thermal_mass) * (
            power_loss - (self.state.motor_temperature - cfg.ambient_temperature) / cfg.thermal_resistance
        )
        self.state.motor_temperature += dT_dt * dt
        
        # --- Power and Energy ---
        self.state.instantaneous_power = supply_voltage * self.state.motor_current
        self.state.cumulative_energy += self.state.instantaneous_power * dt
        
        # --- Degradation ---
        self._update_degradation(dt)
        
        return self.state
    
    def _compute_displacement(self) -> float:
        """Compute diaphragm displacement from crank angle."""
        # Simple harmonic motion: x = (S/2)(1 - cos(θ))
        return (self.config.stroke_length / 2) * (1 - np.cos(self.state.crank_angle))
    
    def _compute_compression_torque(self, inlet_pressure: float) -> float:
        """Compute torque required for compression."""
        displacement = self._compute_displacement()
        current_volume = self._total_volume - self.config.diaphragm_area * displacement
        
        if current_volume <= 0:
            return 0.0
        
        # Pressure force on diaphragm
        compression_ratio = self._total_volume / current_volume
        cylinder_pressure = inlet_pressure * (compression_ratio ** self.config.polytropic_n)
        pressure_force = (cylinder_pressure - inlet_pressure) * self.config.diaphragm_area
        
        # Convert to torque through crank mechanism
        # Simplified: assume crank radius = stroke/2
        crank_radius = self.config.stroke_length / 2
        torque = pressure_force * crank_radius * np.abs(np.sin(self.state.crank_angle))
        
        return max(0, torque)
    
    def _update_degradation(self, dt: float) -> None:
        """Update mechanical wear and health index."""
        # Archard wear model
        # W = K * F * v / H
        # Simplified: wear proportional to speed and cycles
        
        wear_rate = (
            self.config.wear_coefficient * 
            self.state.angular_velocity * 
            (1 + self.state.motor_temperature / 100)  # Temperature acceleration
        )
        
        # Health decreases with wear
        health_decrease = wear_rate * dt / 1e6  # Scale factor
        self.state.health_index = max(0, self.state.health_index - health_decrease)
        
        # Degradation affects efficiency
        # (implemented in compression efficiency if health_index < 0.8)
    
    def get_efficiency(self) -> float:
        """Compute current isentropic efficiency."""
        if self.state.outlet_pressure <= 1.01325e5:
            return 0.0
        
        # Theoretical isentropic work
        gamma = self.config.gamma
        P1 = 1.01325e5
        P2 = self.state.outlet_pressure
        T1 = 293.15  # K
        
        if self.state.mass_flow_rate > 0:
            W_isen = (self.state.mass_flow_rate * self.config.R_gas * T1 * 
                     gamma / (gamma - 1) * 
                     ((P2/P1)**((gamma-1)/gamma) - 1))
            
            if self.state.instantaneous_power > 0:
                efficiency = W_isen / self.state.instantaneous_power
                efficiency *= self.state.health_index  # Degradation effect
                return np.clip(efficiency, 0, 1)
        
        return 0.0
    
    def get_state_dict(self) -> dict:
        """Return state as dictionary for logging/AI."""
        return {
            'crank_angle': self.state.crank_angle,
            'angular_velocity': self.state.angular_velocity,
            'cylinder_pressure': self.state.cylinder_pressure,
            'outlet_pressure': self.state.outlet_pressure,
            'motor_temperature': self.state.motor_temperature,
            'motor_current': self.state.motor_current,
            'mass_flow_rate': self.state.mass_flow_rate,
            'volumetric_flow_lpm': self.state.volumetric_flow * 60000,  # Convert to LPM
            'health_index': self.state.health_index,
            'total_cycles': self.state.total_cycles,
            'instantaneous_power': self.state.instantaneous_power,
            'cumulative_energy_wh': self.state.cumulative_energy / 3600,
            'efficiency': self.get_efficiency()
        }
    
    def get_state(self) -> dict:
        """Dashboard-compatible state getter."""
        return {
            'pressure': self.state.outlet_pressure / 1e5,  # Convert to bar
            'temperature': self.state.motor_temperature,
            'efficiency': self.state.health_index,
            'power': self.state.instantaneous_power,
            'total_runtime': self.state.elapsed_time if hasattr(self.state, 'elapsed_time') else self.state.total_cycles * 0.1,
        }

