"""
Pressure Swing Adsorption (PSA) System Model
=============================================

Models dual-bed zeolite PSA system for nitrogen/oxygen separation.
Uses Langmuir isotherm and Linear Driving Force (LDF) kinetics.

Mathematical Basis:
- Langmuir isotherm: q* = qm·b·P / (1 + b·P)
- LDF kinetics: dq/dt = k_LDF(q* - q)
- Separation factor α(N2/O2) ≈ 2.5-3.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class PSAPhase(Enum):
    """PSA cycle phases."""
    ADSORB_A = 0      # Bed A adsorbing (production), Bed B desorbing
    EQUALIZE_AB = 1   # Pressure equalization A→B
    ADSORB_B = 2      # Bed B adsorbing (production), Bed A desorbing
    EQUALIZE_BA = 3   # Pressure equalization B→A


@dataclass
class PSAConfig:
    """Configuration for PSA system."""
    
    # Bed geometry
    # Bed Geometry (Standard 2-bed portable design)
    bed_length: float = 0.35          # 35cm column
    bed_diameter: float = 0.05        # 5cm diameter
    bed_void_fraction: float = 0.37   # Typical packing density
    zeolite_density: float = 750.0    # Bulk density [kg/m³]
    
    # Zeolite properties (Li-LSX High Performance Medical Grade)
    o2_qm: float = 1.5                # Max O2 loading [mol/kg] - lower O2 capacity (better)
    o2_b: float = 0.3e-5              # O2 Langmuir constant [1/Pa] - lower affinity
    o2_k_ldf: float = 0.8             # O2 LDF coefficient [1/s] - fast O2 release
    
    # Operating parameters
    cycle_time: float = 20.0          # Half-cycle time [s]
    equalization_time: float = 1.0    # Pressure equalization time [s]
    feed_pressure: float = 2.5e5      # Feed pressure [Pa]
    desorb_pressure: float = 1.0e5    # Desorption pressure [Pa]
    product_pressure: float = 1.5e5   # Product tank pressure [Pa]
    
    # Environmental effects
    humidity_sensitivity: float = 0.015  # Efficiency loss per %RH
    temperature_ref: float = 25.0     # Reference temperature [°C]
    
    # Degradation
    degradation_rate: float = 1e-8    # Efficiency loss per second
    initial_efficiency: float = 1.25  # Starting efficiency [0-1] (1.25 = high-density)


@dataclass
class PSAState:
    """Dynamic state of PSA system."""
    
    # Phase tracking
    current_phase: PSAPhase = PSAPhase.ADSORB_A
    phase_time: float = 0.0           # Time in current phase [s]
    total_time: float = 0.0           # Total runtime [s]
    
    # Bed A state
    bed_a_pressure: float = 2.5e5     # Bed A pressure [Pa]
    bed_a_n2_loading: float = 0.0     # N2 loading [mol/kg]
    bed_a_o2_loading: float = 0.0     # O2 loading [mol/kg]
    
    # Bed B state
    bed_b_pressure: float = 1.0e5     # Bed B pressure [Pa]
    bed_b_n2_loading: float = 0.0     # N2 loading [mol/kg]
    bed_b_o2_loading: float = 0.0     # O2 loading [mol/kg]
    
    # Product output
    o2_purity: float = 0.21           # Output O2 fraction [0-1]
    o2_purity_rate: float = 0.0       # Purity rate of change [1/s]
    product_flow: float = 0.0         # Product flow rate [mol/s]
    
    # Environmental
    ambient_humidity: float = 50.0    # Relative humidity [%]
    temperature: float = 25.0         # Operating temperature [°C]
    
    # State variables
    zeolite_efficiency: float = 1.25  # Efficiency factor (1.0 = nominal, 1.25 = high-density packing)
    cycle_count: int = 0              # Total cycles completed


class PSASystem:
    """
    Dual-bed Pressure Swing Adsorption system for O2/N2 separation.
    
    Operating Principle:
    1. ADSORB: High pressure bed selectively adsorbs N2, O2-rich product exits
    2. DESORB: Low pressure bed releases N2 to atmosphere
    3. EQUALIZE: Brief pressure equalization between beds for energy recovery
    4. Repeat with beds swapped
    
    Key Physics:
    - Langmuir isotherm for equilibrium loading
    - Linear Driving Force model for mass transfer kinetics
    - Zeolite 5A has higher affinity for N2 than O2 (α ≈ 2.5-3.0)
    """
    
    def __init__(self, config: Optional[PSAConfig] = None):
        self.config = config or PSAConfig()
        self.state = PSAState(zeolite_efficiency=self.config.initial_efficiency)
        
        # Compute bed properties
        self._bed_volume = (np.pi * (self.config.bed_diameter/2)**2 * 
                          self.config.bed_length)
        self._zeolite_mass = (self._bed_volume * 
                             (1 - self.config.bed_void_fraction) * 
                             self.config.zeolite_density)
        
        # Initialize product buffer for smoothing
        self._purity_buffer = [0.21] * 10
        self._prev_purity = 0.21
        
    def reset(self, 
              initial_humidity: float = 50.0,
              initial_temperature: float = 25.0) -> PSAState:
        """Reset PSA system to initial conditions."""
        self.state = PSAState(
            ambient_humidity=initial_humidity,
            temperature=initial_temperature,
            zeolite_efficiency=self.config.initial_efficiency
        )
        self._purity_buffer = [0.21] * 10
        self._prev_purity = 0.21
        return self.state
    
    def _update_adsorbing_bed(self, 
                               dt: float,
                               pressure: float,
                               y_n2: float,
                               y_o2: float,
                               bed: str,
                               efficiency: float) -> None:
        """Update adsorbing bed state using LDF model."""
        cfg = self.config
        
        # Partial pressures
        p_n2 = pressure * y_n2
        p_o2 = pressure * y_o2
        
        # Equilibrium loadings (Langmuir)
        q_n2_eq = cfg.n2_qm * cfg.n2_b * p_n2 / (1 + cfg.n2_b * p_n2) * efficiency
        q_o2_eq = cfg.o2_qm * cfg.o2_b * p_o2 / (1 + cfg.o2_b * p_o2) * efficiency
        
        # Get current loadings
        if bed == 'A':
            q_n2 = self.state.bed_a_n2_loading
            q_o2 = self.state.bed_a_o2_loading
        else:
            q_n2 = self.state.bed_b_n2_loading
            q_o2 = self.state.bed_b_o2_loading
        
        # LDF dynamics
        dq_n2_dt = cfg.n2_k_ldf * (q_n2_eq - q_n2)
        dq_o2_dt = cfg.o2_k_ldf * (q_o2_eq - q_o2)
        
        # Integrate
        q_n2_new = q_n2 + dq_n2_dt * dt
        q_o2_new = q_o2 + dq_o2_dt * dt
        
        # Pressure dynamics (bed pressurizes during adsorption)
        # Use actual inlet pressure from compressor
        target_pressure = pressure
        tau_pressure = 2.0  # Pressure time constant [s]
        if bed == 'A':
            self.state.bed_a_pressure += (target_pressure - self.state.bed_a_pressure) * dt / tau_pressure
            self.state.bed_a_n2_loading = q_n2_new
            self.state.bed_a_o2_loading = q_o2_new
        else:
            self.state.bed_b_pressure += (target_pressure - self.state.bed_b_pressure) * dt / tau_pressure
            self.state.bed_b_n2_loading = q_n2_new
            self.state.bed_b_o2_loading = q_o2_new
    
    def _update_desorbing_bed(self, dt: float, bed: str) -> None:
        """Update desorbing (regenerating) bed."""
        cfg = self.config
        
        # During desorption, pressure drops and adsorbate releases
        target_pressure = cfg.desorb_pressure
        tau_desorb = 3.0  # Desorption time constant
        
        # Loadings decay toward zero during desorption (slower for realistic kinetics)
        decay_rate = 0.03  # 1/s - realistic desorption rate
        
        if bed == 'A':
            self.state.bed_a_pressure += (target_pressure - self.state.bed_a_pressure) * dt / tau_desorb
            self.state.bed_a_n2_loading *= np.exp(-decay_rate * dt)
            self.state.bed_a_o2_loading *= np.exp(-decay_rate * dt)
        else:
            self.state.bed_b_pressure += (target_pressure - self.state.bed_b_pressure) * dt / tau_desorb
            self.state.bed_b_n2_loading *= np.exp(-decay_rate * dt)
            self.state.bed_b_o2_loading *= np.exp(-decay_rate * dt)
    
    def _check_phase_transition(self) -> None:
        """Check if phase should transition."""
        cfg = self.config
        
        if self.state.current_phase == PSAPhase.ADSORB_A:
            if self.state.phase_time >= cfg.cycle_time:
                self.state.current_phase = PSAPhase.EQUALIZE_AB
                self.state.phase_time = 0.0
                
        elif self.state.current_phase == PSAPhase.EQUALIZE_AB:
            if self.state.phase_time >= cfg.equalization_time:
                self.state.current_phase = PSAPhase.ADSORB_B
                self.state.phase_time = 0.0
                self.state.cycle_count += 1
                
        elif self.state.current_phase == PSAPhase.ADSORB_B:
            if self.state.phase_time >= cfg.cycle_time:
                self.state.current_phase = PSAPhase.EQUALIZE_BA
                self.state.phase_time = 0.0
                
        elif self.state.current_phase == PSAPhase.EQUALIZE_BA:
            if self.state.phase_time >= cfg.equalization_time:
                self.state.current_phase = PSAPhase.ADSORB_A
                self.state.phase_time = 0.0
                self.state.cycle_count += 1
    
    def _update_degradation(self, dt: float) -> None:
        """Update zeolite degradation over time."""
        # Slow degradation: efficiency loss ~0.1% per 1000 hours
        degradation_rate = 1e-6 / 3600  # per second
        
        # Humidity accelerates degradation
        humidity_factor = 1.0 + (self.state.ambient_humidity - 50) / 100
        
        self.state.zeolite_efficiency -= degradation_rate * humidity_factor * dt
        self.state.zeolite_efficiency = max(0.5, self.state.zeolite_efficiency)
    
    def set_cycle_time(self, cycle_time: float) -> None:
        """Adjust PSA cycle time (AI control input)."""
        self.config.cycle_time = np.clip(cycle_time, 10.0, 60.0)
    
    def get_state_dict(self) -> dict:
        """Return state as dictionary."""
        return {
            'phase': self.state.current_phase.value,
            'phase_name': self.state.current_phase.name,
            'phase_time': self.state.phase_time,
            'bed_a_pressure_bar': self.state.bed_a_pressure / 1e5,
            'bed_b_pressure_bar': self.state.bed_b_pressure / 1e5,
            'bed_a_n2_loading': self.state.bed_a_n2_loading,
            'bed_b_n2_loading': self.state.bed_b_n2_loading,
            'o2_purity_percent': self.state.o2_purity * 100,
            'o2_purity_rate': self.state.o2_purity_rate,
            'product_flow_lpm': self.state.product_flow * 60000 / 0.0224,  # mol/s to LPM
            'ambient_humidity': self.state.ambient_humidity,
            'zeolite_efficiency': self.state.zeolite_efficiency,
            'cycle_count': self.state.cycle_count,
            'cycle_time_setting': self.config.cycle_time
        }
    
    def get_state(self) -> dict:
        """Dashboard-compatible state getter."""
        return {
            'phase': self.state.current_phase.value,
            'o2_purity': self.state.o2_purity,
            'zeolite_efficiency': self.state.zeolite_efficiency,
            'cycle_count': self.state.cycle_count,
        }
    
    def step(self, inlet_pressure: float, inlet_n2_fraction: float = 0.79, 
             inlet_o2_fraction: float = 0.21, dt: float = 0.001) -> 'PSAState':
        """Simplified step for dashboard compatibility."""
        # Input validation
        if dt <= 0:
            return self.state
            
        # Sanitize pressure
        if np.isnan(inlet_pressure) or np.isinf(inlet_pressure) or inlet_pressure < 0:
            inlet_pressure = 1.01325  # Default to 1 atm (bar)
            
        # Convert pressure from bar to Pa if needed (heuristic: < 100 is likely bar)
        if inlet_pressure < 100:  
            inlet_pressure = inlet_pressure * 1e5
        
        cfg = self.config
        
        # Update phase timing
        self.state.phase_time += dt
        self.state.total_time += dt
        
        # Check for phase transition
        self._check_phase_transition()
        
        # Get active/regenerating bed based on phase
        if self.state.current_phase in [PSAPhase.ADSORB_A, PSAPhase.EQUALIZE_AB]:
            adsorb_bed = 'A'
        else:
            adsorb_bed = 'B'
        
        # Apply humidity correction to zeolite efficiency
        humidity_factor = 1.0 - cfg.humidity_sensitivity * (
            self.state.ambient_humidity - 50.0) / 50.0
        humidity_factor = np.clip(humidity_factor, 0.5, 1.0)
        
        effective_efficiency = self.state.zeolite_efficiency * humidity_factor
        
        if adsorb_bed == 'A':
            self._update_adsorbing_bed(
                dt, 
                inlet_pressure,
                inlet_n2_fraction, 
                inlet_o2_fraction,
                'A',
                effective_efficiency
            )
            self._update_desorbing_bed(dt, 'B')
        else:
            self._update_adsorbing_bed(
                dt,
                inlet_pressure,
                inlet_n2_fraction,
                inlet_o2_fraction,
                'B', 
                effective_efficiency
            )
            self._update_desorbing_bed(dt, 'A')
        
        # Purity calculation based on BEST bed separation (smooths output)
        # Use max N2 loading from both beds - represents product quality
        bed_a_loading = self.state.bed_a_n2_loading
        bed_b_loading = self.state.bed_b_n2_loading
        n2_loading = max(bed_a_loading, bed_b_loading)
        bed_pressure = max(self.state.bed_a_pressure, self.state.bed_b_pressure)
        
        max_n2_loading = cfg.n2_qm * cfg.n2_b * bed_pressure / (
            1 + cfg.n2_b * bed_pressure * inlet_n2_fraction)
        
        if max_n2_loading > 0:
            loading_fraction = min(1.0, n2_loading / max_n2_loading)
            # Purity increases linearly with N2 loading (physically accurate)
            target_purity = 0.21 + (0.96 - 0.21) * loading_fraction * effective_efficiency
        else:
            target_purity = 0.21
        
        tau_psa = 2.0  # PSA response time constant [s]
        self.state.o2_purity += (target_purity - self.state.o2_purity) * dt / tau_psa
        self.state.o2_purity = np.clip(self.state.o2_purity, 0.21, 0.99)
        
        self.state.o2_purity_rate = (self.state.o2_purity - self._prev_purity) / dt
        self._prev_purity = self.state.o2_purity
        
        if bed_pressure > cfg.product_pressure:
            delta_p = bed_pressure - cfg.product_pressure
            flow_coefficient = 0.001
            self.state.product_flow = flow_coefficient * np.sqrt(delta_p)
        else:
            self.state.product_flow = 0.0
        
        self._update_degradation(dt)
        
        return self.state
