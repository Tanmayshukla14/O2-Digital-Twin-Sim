"""
Safety Watchdog Module
======================

Deterministic safety layer that enforces hard constraints.
AI recommendations MUST pass through this layer before execution.

Safety Invariants:
- Pressure: 0.5 bar ≤ P ≤ 3.0 bar
- Purity: O2 ≥ 87%
- Temperature: T < 85°C
- Flow: 0 ≤ Q ≤ 10 LPM
- Actuator rate: |dPWM/dt| ≤ 10%/100ms

This is a DETERMINISTIC module - no learning, no adaptation.
All decisions are based on hard-coded safety rules.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto


class SafetyLevel(Enum):
    """Safety status levels."""
    NORMAL = auto()       # All parameters within limits
    WARNING = auto()      # Approaching limits
    ALARM = auto()        # At limits, intervention active
    CRITICAL = auto()     # Emergency shutdown required
    SHUTDOWN = auto()     # System shutdown


class ConstraintType(Enum):
    """Types of safety constraints."""
    PRESSURE_HIGH = auto()
    PRESSURE_LOW = auto()
    PURITY_LOW = auto()
    TEMPERATURE_HIGH = auto()
    FLOW_HIGH = auto()
    ACTUATOR_RATE = auto()
    SENSOR_FAULT = auto()
    COMMUNICATION = auto()


@dataclass
class SafetyLimits:
    """Safety limit definitions with hysteresis."""
    
    # Pressure limits [bar]
    pressure_min_alarm: float = 0.5
    pressure_min_warn: float = 0.7
    pressure_max_warn: float = 2.8
    pressure_max_alarm: float = 3.0
    pressure_emergency_max: float = 3.5  # Hardware relief valve
    
    # Purity limits [fraction 0-1]
    purity_min_alarm: float = 0.87
    purity_min_warn: float = 0.89
    
    # Temperature limits [°C]
    temp_max_warn: float = 75.0
    temp_max_alarm: float = 85.0
    temp_max_shutdown: float = 95.0
    
    # Flow limits [LPM]
    flow_max_alarm: float = 10.0
    flow_max_warn: float = 9.0
    
    # Actuator rate limits
    pwm_rate_max: float = 0.1  # 10% per 100ms
    
    # Timing
    alarm_persistence_time: float = 5.0  # Seconds alarm must persist
    recovery_hysteresis: float = 0.05    # Hysteresis for recovery


@dataclass
class SafetyState:
    """Current safety system state."""
    
    level: SafetyLevel = SafetyLevel.NORMAL
    active_constraints: List[ConstraintType] = field(default_factory=list)
    constraint_messages: List[str] = field(default_factory=list)
    
    # Timing
    alarm_start_time: float = 0.0
    last_update_time: float = 0.0
    
    # Emergency state
    emergency_stop_active: bool = False
    manual_override_active: bool = False
    
    # Actuator state
    last_pwm: float = 0.0
    last_pwm_time: float = 0.0
    
    # Statistics
    total_alarms: int = 0
    total_interventions: int = 0


@dataclass
class SafetyStatus:
    """Status returned after safety check."""
    
    safe: bool
    level: SafetyLevel
    constraints_violated: List[ConstraintType]
    messages: List[str]
    corrected_command: Dict[str, float]
    intervention_active: bool


class SafetyWatchdog:
    """
    Deterministic safety watchdog for oxygen concentrator.
    
    Key Principles:
    1. NEVER trust AI outputs directly
    2. All commands pass through validation
    3. Hard limits are absolute - no exceptions
    4. Fail-safe: on any uncertainty, reduce output
    5. All decisions are auditable and deterministic
    
    DESIGN PHILOSOPHY:
    - Fail-Safe: If in doubt, shut down or vent.
    - Hardware First: Software limits are slightly tighter than hardware relief valves.
    - Deterministic: No 'AI' in this module. Simple IF/THEN logic only.
    - Human Override: Emergency stop always takes precedence.
    """
    
    def __init__(self, limits: Optional[SafetyLimits] = None):
        self.limits = limits or SafetyLimits()
        self.state = SafetyState()
        
        # Calibration offset (simulated sensor drift)
        self._pressure_offset = 0.0
        
        # Watchdog timer
        self._last_heartbeat = 0.0
        self._heartbeat_timeout = 0.5  # 500ms
        
    def reset(self) -> None:
        """Reset safety state."""
        self.state = SafetyState()
        self._last_heartbeat = 0.0
    
    def validate_command(self,
                         command: Dict[str, float],
                         current_state: Dict[str, float],
                         current_time: float) -> SafetyStatus:
        """
        Validate and potentially modify a control command.
        
        Args:
            command: Proposed control outputs
                - 'pwm_duty': Motor PWM (0-1)
                - 'flow_setpoint': Target flow (LPM)
                - 'pressure_setpoint': Target pressure (bar)
            current_state: Current sensor readings
                - 'pressure': Tank pressure (bar)
                - 'purity': O2 purity (fraction)
                - 'temperature': Motor temp (°C)
                - 'flow': Current flow (LPM)
            current_time: Current simulation time [s]
            
        Returns:
            SafetyStatus with potentially corrected command
        """
        if command is None:
            command = {}
            
        self.state.last_update_time = current_time
        
        violations = []
        messages = []
        corrected = command.copy()
        intervention = False
        
        lim = self.limits
        
        # === Pressure Checks ===
        pressure = current_state.get('pressure', 0)
        
        if pressure >= lim.pressure_max_alarm:
            violations.append(ConstraintType.PRESSURE_HIGH)
            messages.append(f"ALARM: Pressure {pressure:.2f} bar >= {lim.pressure_max_alarm}")
            # Reduce PWM to lower pressure
            corrected['pwm_duty'] = min(corrected.get('pwm_duty', 0), 0.3)
            intervention = True
            
        elif pressure >= lim.pressure_max_warn:
            messages.append(f"WARN: Pressure {pressure:.2f} bar approaching limit")
            # Limit PWM increase
            corrected['pwm_duty'] = min(corrected.get('pwm_duty', 0), 0.7)
        
        if pressure <= lim.pressure_min_alarm:
            violations.append(ConstraintType.PRESSURE_LOW)
            messages.append(f"ALARM: Pressure {pressure:.2f} bar <= {lim.pressure_min_alarm}")
            # Increase PWM to raise pressure
            corrected['pwm_duty'] = max(corrected.get('pwm_duty', 0), 0.8)
            intervention = True
            
        elif pressure <= lim.pressure_min_warn:
            messages.append(f"WARN: Pressure {pressure:.2f} bar low")
        
        # === Purity Checks ===
        purity = current_state.get('purity', 1.0)
        
        if purity < lim.purity_min_alarm:
            violations.append(ConstraintType.PURITY_LOW)
            messages.append(f"ALARM: Purity {purity*100:.1f}% < {lim.purity_min_alarm*100}%")
            # Cannot directly fix purity, but flag for PSA adjustment
            corrected['purity_alarm'] = True
            intervention = True
            
        elif purity < lim.purity_min_warn:
            messages.append(f"WARN: Purity {purity*100:.1f}% low")
        
        # === Temperature Checks ===
        temperature = current_state.get('temperature', 25)
        
        if temperature >= lim.temp_max_shutdown:
            violations.append(ConstraintType.TEMPERATURE_HIGH)
            messages.append(f"CRITICAL: Temperature {temperature:.1f}°C - SHUTDOWN")
            corrected['pwm_duty'] = 0.0
            corrected['emergency_stop'] = True
            self.state.emergency_stop_active = True
            intervention = True
            
        elif temperature >= lim.temp_max_alarm:
            violations.append(ConstraintType.TEMPERATURE_HIGH)
            messages.append(f"ALARM: Temperature {temperature:.1f}°C >= {lim.temp_max_alarm}")
            corrected['pwm_duty'] = min(corrected.get('pwm_duty', 0), 0.5)
            intervention = True
            
        elif temperature >= lim.temp_max_warn:
            messages.append(f"WARN: Temperature {temperature:.1f}°C high")
            corrected['pwm_duty'] = min(corrected.get('pwm_duty', 0), 0.8)
        
        # === Flow Checks ===
        flow_setpoint = corrected.get('flow_setpoint', 0)
        
        if flow_setpoint > lim.flow_max_alarm:
            violations.append(ConstraintType.FLOW_HIGH)
            messages.append(f"ALARM: Flow setpoint {flow_setpoint:.1f} LPM > max")
            corrected['flow_setpoint'] = lim.flow_max_alarm
            intervention = True
            
        elif flow_setpoint > lim.flow_max_warn:
            messages.append(f"WARN: Flow setpoint {flow_setpoint:.1f} LPM high")
        
        # === Actuator Rate Limiting ===
        pwm_requested = corrected.get('pwm_duty', 0)
        dt = current_time - self.state.last_pwm_time if self.state.last_pwm_time > 0 else 0.1
        
        if dt > 0:
            pwm_rate = abs(pwm_requested - self.state.last_pwm) / dt
            max_rate = lim.pwm_rate_max / 0.1  # Convert to per-second
            
            if pwm_rate > max_rate:
                violations.append(ConstraintType.ACTUATOR_RATE)
                allowed_change = np.sign(pwm_requested - self.state.last_pwm) * max_rate * dt
                corrected['pwm_duty'] = self.state.last_pwm + allowed_change
                messages.append(f"Rate limited PWM: {self.state.last_pwm:.2f} -> {corrected['pwm_duty']:.2f}")
        
        # Update PWM tracking
        self.state.last_pwm = corrected.get('pwm_duty', 0)
        self.state.last_pwm_time = current_time
        
        # === Clamp All Outputs ===
        corrected['pwm_duty'] = np.clip(corrected.get('pwm_duty', 0), 0.0, 1.0)
        corrected['flow_setpoint'] = np.clip(
            corrected.get('flow_setpoint', 0), 0.0, lim.flow_max_alarm)
        corrected['pressure_setpoint'] = np.clip(
            corrected.get('pressure_setpoint', 2.0), 
            lim.pressure_min_alarm, 
            lim.pressure_max_alarm
        )
        
        # === Update Safety Level ===
        if self.state.emergency_stop_active:
            level = SafetyLevel.SHUTDOWN
        elif any(v in [ConstraintType.TEMPERATURE_HIGH] for v in violations):
            level = SafetyLevel.CRITICAL
        elif len(violations) > 0:
            level = SafetyLevel.ALARM
            self.state.total_alarms += 1
        elif len(messages) > 0:
            level = SafetyLevel.WARNING
        else:
            level = SafetyLevel.NORMAL
        
        self.state.level = level
        self.state.active_constraints = violations
        self.state.constraint_messages = messages
        
        if intervention:
            self.state.total_interventions += 1
        
        return SafetyStatus(
            safe=len(violations) == 0,
            level=level,
            constraints_violated=violations,
            messages=messages,
            corrected_command=corrected,
            intervention_active=intervention
        )
    
    def check_heartbeat(self, current_time: float) -> bool:
        """
        Check watchdog heartbeat.
        
        Must be called regularly. If not called within timeout,
        system assumes controller failure and triggers safe state.
        
        Returns:
            True if heartbeat OK, False if timeout
        """
        dt = current_time - self._last_heartbeat
        self._last_heartbeat = current_time
        
        if dt > self._heartbeat_timeout:
            self.state.level = SafetyLevel.ALARM
            self.state.active_constraints.append(ConstraintType.COMMUNICATION)
            return False
        
        return True
    
    def emergency_stop(self) -> Dict[str, float]:
        """
        Trigger emergency stop.
        
        Returns safe state commands.
        """
        self.state.emergency_stop_active = True
        self.state.level = SafetyLevel.SHUTDOWN
        
        return {
            'pwm_duty': 0.0,
            'flow_setpoint': 0.0,
            'pressure_setpoint': 0.0,
            'emergency_stop': True,
        }
    
    def reset_emergency(self) -> bool:
        """
        Attempt to reset emergency stop.
        
        Returns:
            True if reset successful, False if conditions not safe
        """
        # Only reset if no active critical constraints
        if ConstraintType.TEMPERATURE_HIGH in self.state.active_constraints:
            return False
        
        self.state.emergency_stop_active = False
        self.state.level = SafetyLevel.NORMAL
        return True
    
    def get_status(self) -> SafetyStatus:
        """Get current safety status."""
        return SafetyStatus(
            safe=self.state.level in [SafetyLevel.NORMAL, SafetyLevel.WARNING],
            level=self.state.level,
            constraints_violated=self.state.active_constraints,
            messages=self.state.constraint_messages,
            corrected_command={},
            intervention_active=self.state.emergency_stop_active
        )
    
    def get_telemetry(self) -> Dict:
        """Get safety system telemetry."""
        return {
            'level': self.state.level.name,
            'active_constraints': [c.name for c in self.state.active_constraints],
            'emergency_stop': self.state.emergency_stop_active,
            'total_alarms': self.state.total_alarms,
            'total_interventions': self.state.total_interventions,
            'last_pwm': self.state.last_pwm,
        }
