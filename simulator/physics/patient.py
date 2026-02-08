"""
Patient Oxygen Demand Model
===========================

Models patient SpO2 response to supplemental oxygen therapy.
Includes respiratory dynamics, metabolic demand, and clinical scenarios.

Mathematical Basis:
- SpO2 first-order response: dSpO2/dt = (1/τ)(SpO2_target - SpO2)
- Oxygen delivery: DO2 = Qc × CaO2
- Metabolic consumption: VO2 = VO2_basal × activity_factor
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class PatientCondition(Enum):
    """Clinical patient conditions affecting O2 requirements."""
    HEALTHY = 0          # Healthy individual (baseline)
    COPD_MILD = 1        # Mild COPD
    COPD_MODERATE = 2    # Moderate COPD
    COPD_SEVERE = 3      # Severe COPD
    PNEUMONIA = 4        # Acute pneumonia
    POST_SURGICAL = 5    # Post-operative patient
    SLEEP = 6            # Sleeping patient
    EXERCISE = 7         # Light exercise/activity


class ActivityLevel(Enum):
    """Patient activity levels."""
    RESTING = 0
    LIGHT = 1
    MODERATE = 2
    ACTIVE = 3


@dataclass
class PatientConfig:
    """Patient model configuration."""
    
    # Baseline physiology (healthy adult)
    basal_spo2: float = 98.0              # Baseline SpO2 [%]
    basal_respiratory_rate: float = 14.0   # Breaths per minute
    basal_tidal_volume: float = 0.5        # Tidal volume [L]
    basal_vo2: float = 250.0              # Basal O2 consumption [mL/min]
    
    # SpO2 dynamics
    spo2_time_constant: float = 45.0      # SpO2 response time constant [s]
    spo2_noise_std: float = 0.5           # SpO2 measurement noise [%]
    
    # Hemoglobin
    hemoglobin: float = 14.0              # Hemoglobin [g/dL]
    
    # Condition-specific parameters
    condition: PatientCondition = PatientCondition.COPD_MODERATE
    activity: ActivityLevel = ActivityLevel.RESTING
    
    # Clinical targets
    target_spo2_min: float = 88.0         # Minimum acceptable SpO2
    target_spo2_max: float = 96.0         # Maximum target SpO2	(avoid over-oxygenation)
    
    # Environmental
    altitude_m: float = 0.0               # Altitude above sea level [m]


@dataclass
class PatientState:
    """Dynamic patient state."""
    
    # Vital signs
    spo2: float = 94.0                    # Oxygen saturation [%]
    spo2_trend: float = 0.0               # SpO2 derivative [%/min]
    respiratory_rate: float = 16.0        # Breaths per minute
    heart_rate: float = 75.0              # Beats per minute
    
    # Oxygen parameters
    fio2: float = 0.21                    # Fraction inspired O2
    o2_flow_received: float = 0.0         # O2 flow received [LPM]
    vo2: float = 250.0                    # Current O2 consumption [mL/min]
    
    # Derived values
    pao2_estimated: float = 80.0          # Estimated arterial PO2 [mmHg]
    oxygen_deficit: float = 0.0           # O2 supply-demand gap [mL/min]
    
    # Clinical status
    desaturation_duration: float = 0.0    # Time SpO2 < 88% [s]
    hypoxia_events: int = 0               # Count of hypoxia episodes
    
    # Time
    elapsed_time: float = 0.0             # Total simulation time [s]


class PatientModel:
    """
    Patient oxygen physiology model.
    
    Models the relationship between supplemental oxygen delivery
    and patient SpO2 response, accounting for:
    - Disease condition (COPD severity, pneumonia, etc.)
    - Activity level and metabolic demand
    - Respiratory mechanics
    - Altitude effects
    
    The model is clinically conservative, erring on the side
    of patient safety (faster desaturation, slower recovery).
    """
    
    # Condition-specific multipliers
    CONDITION_PARAMS = {
        PatientCondition.HEALTHY: {
            'baseline_spo2': 98.0,
            'o2_sensitivity': 1.0,
            'recovery_rate': 1.0,
            'desaturation_rate': 0.5,
            'min_achievable_spo2': 96.0,
        },
        PatientCondition.COPD_MILD: {
            'baseline_spo2': 94.0,
            'o2_sensitivity': 0.8,
            'recovery_rate': 0.7,
            'desaturation_rate': 1.0,
            'min_achievable_spo2': 88.0,
        },
        PatientCondition.COPD_MODERATE: {
            'baseline_spo2': 91.0,
            'o2_sensitivity': 0.6,
            'recovery_rate': 0.5,
            'desaturation_rate': 1.5,
            'min_achievable_spo2': 85.0,
        },
        PatientCondition.COPD_SEVERE: {
            'baseline_spo2': 88.0,
            'o2_sensitivity': 0.4,
            'recovery_rate': 0.3,
            'desaturation_rate': 2.0,
            'min_achievable_spo2': 80.0,
        },
        PatientCondition.PNEUMONIA: {
            'baseline_spo2': 90.0,
            'o2_sensitivity': 0.7,
            'recovery_rate': 0.4,
            'desaturation_rate': 1.8,
            'min_achievable_spo2': 82.0,
        },
        PatientCondition.POST_SURGICAL: {
            'baseline_spo2': 93.0,
            'o2_sensitivity': 0.85,
            'recovery_rate': 0.6,
            'desaturation_rate': 1.2,
            'min_achievable_spo2': 88.0,
        },
        PatientCondition.SLEEP: {
            'baseline_spo2': 94.0,
            'o2_sensitivity': 0.9,
            'recovery_rate': 0.8,
            'desaturation_rate': 0.8,
            'min_achievable_spo2': 90.0,
        },
        PatientCondition.EXERCISE: {
            'baseline_spo2': 96.0,
            'o2_sensitivity': 1.0,
            'recovery_rate': 0.9,
            'desaturation_rate': 1.5,
            'min_achievable_spo2': 88.0,
        },
    }
    
    ACTIVITY_VO2_MULTIPLIER = {
        ActivityLevel.RESTING: 1.0,
        ActivityLevel.LIGHT: 1.5,
        ActivityLevel.MODERATE: 2.5,
        ActivityLevel.ACTIVE: 4.0,
    }
    
    def __init__(self, config: Optional[PatientConfig] = None):
        self.config = config or PatientConfig()
        self.state = PatientState()
        
        # Initialize based on condition
        params = self.CONDITION_PARAMS[self.config.condition]
        self.state.spo2 = params['baseline_spo2']
        
        self._prev_spo2 = self.state.spo2
        self._trend_buffer = [0.0] * 10
        
    def reset(self, 
              condition: Optional[PatientCondition] = None,
              activity: Optional[ActivityLevel] = None) -> PatientState:
        """Reset patient to initial conditions."""
        if condition:
            self.config.condition = condition
        if activity:
            self.config.activity = activity
            
        params = self.CONDITION_PARAMS[self.config.condition]
        self.state = PatientState(spo2=params['baseline_spo2'])
        self._prev_spo2 = self.state.spo2
        self._trend_buffer = [0.0] * 10
        
        return self.state
    
    def _step_internal(self, dt: float, o2_flow_lpm: float, o2_purity: float) -> PatientState:
        """
        Internal step implementation - called by public step() method.
        """
        pass  # This method is replaced by the dashboard-compatible step() below
    
    @property
    def _prev_above_88(self) -> bool:
        return getattr(self, '_was_above_88', True)
    
    @_prev_above_88.setter
    def _prev_above_88(self, value: bool):
        self._was_above_88 = value
    
    def set_condition(self, condition: PatientCondition) -> None:
        """Change patient condition (for scenario testing)."""
        self.config.condition = condition
    
    def set_activity(self, activity: ActivityLevel) -> None:
        """Change activity level."""
        self.config.activity = activity
    
    def get_o2_requirement(self) -> float:
        """Estimate O2 flow needed to maintain target SpO2."""
        params = self.CONDITION_PARAMS[self.config.condition]
        target = (self.config.target_spo2_min + self.config.target_spo2_max) / 2
        
        if self.state.spo2 >= target:
            return max(0.5, self.state.o2_flow_received - 0.5)
        else:
            deficit = target - self.state.spo2
            return min(10.0, self.state.o2_flow_received + deficit * 0.2)
    
    def get_state_dict(self) -> dict:
        """Return state as dictionary for logging/AI."""
        return {
            'spo2': self.state.spo2,
            'spo2_trend': self.state.spo2_trend,
            'fio2': self.state.fio2,
            'o2_flow_lpm': self.state.o2_flow_received,
            'vo2_ml_min': self.state.vo2,
            'pao2_estimated': self.state.pao2_estimated,
            'oxygen_deficit': self.state.oxygen_deficit,
            'respiratory_rate': self.state.respiratory_rate,
            'heart_rate': self.state.heart_rate,
            'desaturation_duration_s': self.state.desaturation_duration,
            'hypoxia_events': self.state.hypoxia_events,
            'condition': self.config.condition.name,
            'activity': self.config.activity.name,
            'elapsed_time_s': self.state.elapsed_time
        }
    
    def get_state(self) -> dict:
        """Dashboard-compatible state getter."""
        return {
            'spo2': self.state.spo2,
            'spo2_trend': self.state.spo2_trend,
            'activity_level': self.ACTIVITY_VO2_MULTIPLIER.get(self.config.activity, 1.0),
            'oxygen_deficit': self.state.oxygen_deficit,
            'respiratory_rate': self.state.respiratory_rate,
        }
    
    def step(self, delivered_o2_lpm: float, o2_purity: float, dt: float = 0.001) -> 'PatientState':
        """Dashboard-compatible step using simpler signature."""
        self.state.elapsed_time += dt
        self.state.o2_flow_received = delivered_o2_lpm
        
        params = self.CONDITION_PARAMS[self.config.condition]
        
        # Calculate FiO2
        minute_ventilation = (self.config.basal_respiratory_rate * 
                             self.config.basal_tidal_volume)
        
        effective_o2_flow = delivered_o2_lpm * o2_purity
        room_air_flow = max(0, minute_ventilation - delivered_o2_lpm)
        
        total_o2 = effective_o2_flow + room_air_flow * 0.21
        self.state.fio2 = min(1.0, total_o2 / minute_ventilation)
        
        # Calculate Target SpO2
        fio2_effect = (self.state.fio2 - 0.21) / 0.79
        
        min_spo2 = params['min_achievable_spo2']
        max_spo2 = min(99.0, params['baseline_spo2'] + 6.0)
        
        target_spo2 = min_spo2 + (max_spo2 - min_spo2) * fio2_effect * params['o2_sensitivity']
        
        altitude_factor = np.exp(-self.config.altitude_m / 8500)
        target_spo2 *= altitude_factor
        
        # Metabolic Demand
        activity_mult = self.ACTIVITY_VO2_MULTIPLIER[self.config.activity]
        self.state.vo2 = self.config.basal_vo2 * activity_mult
        
        if activity_mult > 1.0:
            target_spo2 -= (activity_mult - 1.0) * 2.0
        
        # SpO2 Dynamics
        if target_spo2 > self.state.spo2:
            tau = self.config.spo2_time_constant / params['recovery_rate']
        else:
            tau = self.config.spo2_time_constant / params['desaturation_rate']
        
        dspo2_dt = (target_spo2 - self.state.spo2) / tau
        self.state.spo2 += dspo2_dt * dt
        self.state.spo2 = np.clip(self.state.spo2, 50.0, 100.0)
        
        # Calculate Trend
        instant_trend = (self.state.spo2 - self._prev_spo2) / dt * 60
        self._trend_buffer.pop(0)
        self._trend_buffer.append(instant_trend)
        self.state.spo2_trend = np.mean(self._trend_buffer)
        self._prev_spo2 = self.state.spo2
        
        # Derived Values
        p50 = 26.6
        n = 2.7
        
        spo2_frac = self.state.spo2 / 100.0
        if spo2_frac < 0.99 and spo2_frac > 0.01:
            self.state.pao2_estimated = p50 * ((spo2_frac / (1 - spo2_frac)) ** (1/n))
            self.state.pao2_estimated = np.clip(self.state.pao2_estimated, 20, 500)
        
        cardiac_output = 5.0
        cao2 = (1.34 * self.config.hemoglobin * spo2_frac + 
                0.003 * self.state.pao2_estimated)
        do2 = cardiac_output * cao2 * 10
        
        self.state.oxygen_deficit = max(0, self.state.vo2 - do2 * 0.25)
        
        # Clinical Monitoring
        if self.state.spo2 < 88.0:
            self.state.desaturation_duration += dt
            if self.state.desaturation_duration >= 30 and self._prev_above_88:
                self.state.hypoxia_events += 1
                self._prev_above_88 = False
        else:
            self.state.desaturation_duration = 0.0
            self._prev_above_88 = True
        
        # Physiological Variation
        if self.state.spo2 < 90:
            self.state.respiratory_rate = self.config.basal_respiratory_rate * (1.5 - self.state.spo2/100)
        else:
            self.state.respiratory_rate = self.config.basal_respiratory_rate
        
        if self.state.spo2 < 88:
            self.state.heart_rate = 75 + (88 - self.state.spo2) * 2
        else:
            self.state.heart_rate = 75
        
        return self.state
