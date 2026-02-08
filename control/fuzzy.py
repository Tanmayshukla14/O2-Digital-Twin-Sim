"""
Fuzzy Logic Oxygen Titration Controller
========================================

Clinical fuzzy controller for automatic oxygen titration based on:
- Patient SpO2 level and trend
- Oxygen purity status
- Clinical guidelines for LTOT (Long-Term Oxygen Therapy)

Rule base derived from clinical practice guidelines.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


@dataclass
class FuzzyConfig:
    """Fuzzy controller configuration."""
    
    # SpO2 target ranges
    spo2_target_low: float = 88.0         # Minimum acceptable SpO2
    spo2_target_high: float = 94.0        # Target SpO2
    spo2_danger_low: float = 85.0         # Danger zone
    spo2_avoid_high: float = 96.0         # Avoid over-oxygenation
    
    # Output limits
    flow_min: float = 0.5                 # Minimum flow [LPM]
    flow_max: float = 10.0                # Maximum flow [LPM]
    
    # Update rate
    update_interval: float = 10.0         # Seconds between adjustments
    
    # Smoothing
    output_filter_alpha: float = 0.3      # Exponential smoothing factor


class TriangularMF:
    """Triangular membership function."""
    
    def __init__(self, left: float, center: float, right: float):
        self.left = left
        self.center = center
        self.right = right
    
    def __call__(self, x: float) -> float:
        if x <= self.left or x >= self.right:
            return 0.0
        elif x <= self.center:
            return (x - self.left) / (self.center - self.left)
        else:
            return (self.right - x) / (self.right - self.center)


class TrapezoidalMF:
    """Trapezoidal membership function."""
    
    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a  # Left foot
        self.b = b  # Left shoulder
        self.c = c  # Right shoulder
        self.d = d  # Right foot
    
    def __call__(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        elif x <= self.b:
            return (x - self.a) / (self.b - self.a)
        elif x <= self.c:
            return 1.0
        else:
            return (self.d - x) / (self.d - self.c)


class FuzzyOxygenTitrator:
    """
    Fuzzy logic controller for oxygen flow titration.
    
    Inputs:
        - SpO2: Current oxygen saturation (%)
        - SpO2_trend: Rate of change (%/min)
        
    Output:
        - Flow adjustment: Change in O2 flow (LPM)
    
    Rule Base (25 rules):
        Based on clinical oxygen therapy guidelines.
        Rules ensure patient safety while avoiding over-oxygenation.
    """
    
    def __init__(self, config: Optional[FuzzyConfig] = None):
        self.config = config or FuzzyConfig()
        
        # === Input Membership Functions ===
        
        # SpO2 levels
        self.mf_spo2 = {
            'CRITICAL': TrapezoidalMF(0, 0, 80, 85),      # < 85%
            'LOW': TriangularMF(82, 88, 92),              # 85-92%
            'TARGET': TriangularMF(88, 93, 96),           # 88-96%
            'HIGH': TriangularMF(94, 97, 100),            # 94-100%
            'DANGER_HIGH': TrapezoidalMF(96, 98, 100, 100),  # > 96%
        }
        
        # SpO2 trend (%/min)
        self.mf_trend = {
            'FALLING_FAST': TrapezoidalMF(-10, -10, -2, -0.5),  # < -0.5%/min
            'FALLING': TriangularMF(-2, -0.5, 0),                # -2 to 0%/min
            'STABLE': TriangularMF(-0.5, 0, 0.5),               # ±0.5%/min
            'RISING': TriangularMF(0, 0.5, 2),                  # 0 to 2%/min
            'RISING_FAST': TrapezoidalMF(0.5, 2, 10, 10),       # > 0.5%/min
        }
        
        # === Output Membership Functions ===
        # Flow adjustment (LPM)
        self.mf_output = {
            'DECREASE_LARGE': -2.0,    # -2 LPM
            'DECREASE_SMALL': -0.5,    # -0.5 LPM
            'MAINTAIN': 0.0,           # No change
            'INCREASE_SMALL': 0.5,     # +0.5 LPM
            'INCREASE_MEDIUM': 1.0,    # +1 LPM
            'INCREASE_LARGE': 2.0,     # +2 LPM
        }
        
        # === Rule Base ===
        # (spo2_level, trend) -> output
        self.rules: List[Tuple[str, str, str]] = [
            # Critical SpO2 - always increase
            ('CRITICAL', 'FALLING_FAST', 'INCREASE_LARGE'),
            ('CRITICAL', 'FALLING', 'INCREASE_LARGE'),
            ('CRITICAL', 'STABLE', 'INCREASE_LARGE'),
            ('CRITICAL', 'RISING', 'INCREASE_MEDIUM'),
            ('CRITICAL', 'RISING_FAST', 'INCREASE_SMALL'),
            
            # Low SpO2
            ('LOW', 'FALLING_FAST', 'INCREASE_LARGE'),
            ('LOW', 'FALLING', 'INCREASE_MEDIUM'),
            ('LOW', 'STABLE', 'INCREASE_MEDIUM'),
            ('LOW', 'RISING', 'INCREASE_SMALL'),
            ('LOW', 'RISING_FAST', 'MAINTAIN'),
            
            # Target SpO2 (desired range)
            ('TARGET', 'FALLING_FAST', 'INCREASE_MEDIUM'),
            ('TARGET', 'FALLING', 'INCREASE_SMALL'),
            ('TARGET', 'STABLE', 'MAINTAIN'),
            ('TARGET', 'RISING', 'MAINTAIN'),
            ('TARGET', 'RISING_FAST', 'DECREASE_SMALL'),
            
            # High SpO2
            ('HIGH', 'FALLING_FAST', 'MAINTAIN'),
            ('HIGH', 'FALLING', 'MAINTAIN'),
            ('HIGH', 'STABLE', 'DECREASE_SMALL'),
            ('HIGH', 'RISING', 'DECREASE_SMALL'),
            ('HIGH', 'RISING_FAST', 'DECREASE_LARGE'),
            
            # Danger high SpO2 - always decrease (avoid O2 toxicity)
            ('DANGER_HIGH', 'FALLING_FAST', 'MAINTAIN'),
            ('DANGER_HIGH', 'FALLING', 'DECREASE_SMALL'),
            ('DANGER_HIGH', 'STABLE', 'DECREASE_LARGE'),
            ('DANGER_HIGH', 'RISING', 'DECREASE_LARGE'),
            ('DANGER_HIGH', 'RISING_FAST', 'DECREASE_LARGE'),
        ]
        
        # State
        self._prev_output = 0.0
        self._last_update_time = 0.0
        self._current_flow = 3.0  # Starting flow
        
    def reset(self, initial_flow: float = 3.0) -> None:
        """Reset controller state."""
        self._prev_output = 0.0
        self._current_flow = initial_flow
        self._last_update_time = 0.0
    
    def compute(self, 
                spo2: float, 
                spo2_trend: float,
                current_time: float = 0.0) -> Dict[str, float]:
        """
        Compute fuzzy oxygen titration.
        
        Args:
            spo2: Current SpO2 (%)
            spo2_trend: SpO2 rate of change (%/min)
            current_time: Current simulation time [s]
            
        Returns:
            Dict with 'flow_adjustment' and 'recommended_flow'
        """
        # Rate limiting
        if current_time - self._last_update_time < self.config.update_interval:
            return {
                'flow_adjustment': 0.0,
                'recommended_flow': self._current_flow,
                'update_pending': True,
            }
        
        # === Fuzzification ===
        spo2_memberships = {
            name: mf(spo2) for name, mf in self.mf_spo2.items()
        }
        
        trend_memberships = {
            name: mf(spo2_trend) for name, mf in self.mf_trend.items()
        }
        
        # === Rule Evaluation ===
        # Using Mamdani inference with MIN for AND, MAX for aggregation
        output_weights = {name: 0.0 for name in self.mf_output.keys()}
        
        for spo2_term, trend_term, output_term in self.rules:
            # AND = MIN
            rule_strength = min(
                spo2_memberships.get(spo2_term, 0.0),
                trend_memberships.get(trend_term, 0.0)
            )
            
            # MAX aggregation
            output_weights[output_term] = max(
                output_weights[output_term],
                rule_strength
            )
        
        # === Defuzzification ===
        # Weighted average (centroid-like for singletons)
        numerator = 0.0
        denominator = 0.0
        
        for output_term, weight in output_weights.items():
            if weight > 0:
                numerator += weight * self.mf_output[output_term]
                denominator += weight
        
        if denominator > 0:
            flow_adjustment = numerator / denominator
        else:
            flow_adjustment = 0.0
        
        # === Output Smoothing ===
        alpha = self.config.output_filter_alpha
        flow_adjustment = (
            alpha * flow_adjustment + 
            (1 - alpha) * self._prev_output
        )
        self._prev_output = flow_adjustment
        
        # Update flow
        self._current_flow += flow_adjustment
        self._current_flow = np.clip(
            self._current_flow,
            self.config.flow_min,
            self.config.flow_max
        )
        
        self._last_update_time = current_time
        
        return {
            'flow_adjustment': flow_adjustment,
            'recommended_flow': self._current_flow,
            'spo2_memberships': spo2_memberships,
            'trend_memberships': trend_memberships,
            'output_weights': output_weights,
            'update_pending': False,
        }
    
    def get_current_flow(self) -> float:
        """Get current recommended flow."""
        return self._current_flow
    
    def set_flow(self, flow: float) -> None:
        """Set current flow (for external override)."""
        self._current_flow = np.clip(
            flow,
            self.config.flow_min,
            self.config.flow_max
        )
    
    def get_telemetry(self) -> Dict:
        """Get controller telemetry."""
        return {
            'current_flow': self._current_flow,
            'last_adjustment': self._prev_output,
            'last_update_time': self._last_update_time,
        }
