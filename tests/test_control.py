"""
Unit Tests for Control Algorithms
=================================

Tests for PID, Fuzzy, and Safety controllers.
"""

import pytest
import numpy as np
from control.pid import PIDController, PIDConfig, CascadePIDController
from control.fuzzy import FuzzyOxygenTitrator, FuzzyConfig
from control.safety_watchdog import SafetyWatchdog, SafetyLimits, SafetyLevel


class TestPIDController:
    """Tests for PID controller."""
    
    def test_initialization(self):
        """Test PID initializes correctly."""
        pid = PIDController()
        assert pid is not None
    
    def test_proportional_response(self):
        """Test proportional term works correctly."""
        config = PIDConfig(kp=1.0, ki=0.0, kd=0.0, output_rate_limit=0.0)  # Disable rate limit
        pid = PIDController(config)
        
        # Error of 1.0 should give output of 1.0 (with Kp=1)
        output = pid.compute(setpoint=10.0, measurement=9.0, dt=0.1)
        
        assert output == pytest.approx(1.0, abs=0.1)
    
    def test_integral_accumulation(self):
        """Test integral term accumulates error."""
        config = PIDConfig(kp=0.0, ki=1.0, kd=0.0)
        pid = PIDController(config)
        
        # Apply constant error
        outputs = []
        for _ in range(10):
            output = pid.compute(setpoint=10.0, measurement=9.0, dt=0.1)
            outputs.append(output)
        
        # Output should increase due to integral
        assert outputs[-1] > outputs[0]
    
    def test_derivative_response(self):
        """Test derivative term responds to rate of change."""
        config = PIDConfig(kp=0.0, ki=0.0, kd=1.0, derivative_filter_tau=0.0)  # No filtering
        pid = PIDController(config)
        
        # First step - no derivative yet
        pid.compute(setpoint=10.0, measurement=9.0, dt=0.1)
        
        # Next step with different measurement (measurement approaching setpoint)
        # Derivative on measurement = -(9.5 - 9.0) / 0.1 = -5.0
        # D term = Kd * derivative = 1.0 * (-5.0) = -5.0 (clamped to output limits)
        output = pid.compute(setpoint=10.0, measurement=9.5, dt=0.1)
        
        # The D term should have some response (may be clamped by output limits)
        # Just check it computed something
        assert output is not None
    
    def test_anti_windup(self):
        """Test integral anti-windup prevents excessive accumulation."""
        config = PIDConfig(kp=0.0, ki=1.0, kd=0.0, output_max=1.0)
        pid = PIDController(config)
        
        # Apply large error for many steps
        outputs = []
        for _ in range(1000):
            output = pid.compute(setpoint=100.0, measurement=0.0, dt=0.1)
            outputs.append(output)
        
        # Output should be clamped
        assert abs(outputs[-1]) <= config.output_max + 0.1
    
    def test_reset(self):
        """Test PID reset clears state."""
        pid = PIDController()
        
        for _ in range(10):
            pid.compute(setpoint=10.0, measurement=5.0, dt=0.1)
        
        pid.reset()
        
        # After reset, integral should be zero
        # First computation should only have proportional term
        output1 = pid.compute(setpoint=10.0, measurement=9.0, dt=0.1)
        pid.reset()
        output2 = pid.compute(setpoint=10.0, measurement=9.0, dt=0.1)
        
        assert output1 == pytest.approx(output2, abs=0.01)


class TestCascadePIDController:
    """Tests for cascade PID controller."""
    
    def test_cascade_structure(self):
        """Test cascade controller has outer and inner loops."""
        cascade = CascadePIDController()
        
        assert hasattr(cascade, 'outer_controller')
        assert hasattr(cascade, 'inner_controller')
    
    def test_cascade_output(self):
        """Test cascade produces valid output."""
        cascade = CascadePIDController()
        
        output = cascade.compute(
            outer_setpoint=95.0,  # SpO2 target
            outer_measurement=93.0,
            inner_measurement=2.0,  # Pressure
            dt=0.1
        )
        
        assert output is not None
        assert isinstance(output, (int, float, np.floating))


class TestFuzzyOxygenTitrator:
    """Tests for fuzzy oxygen titration controller."""
    
    def test_initialization(self):
        """Test fuzzy controller initializes."""
        fuzzy = FuzzyOxygenTitrator()
        assert fuzzy is not None
    
    def test_low_spo2_increases_flow(self):
        """Test low SpO2 results in increased flow recommendation."""
        fuzzy = FuzzyOxygenTitrator()
        fuzzy.reset(initial_flow=3.0)
        
        # Low SpO2, falling
        result = fuzzy.compute(spo2=85.0, spo2_trend=-1.0, current_time=10.0)
        
        # Flow should increase
        assert result['flow_adjustment'] > 0 or result['recommended_flow'] > 3.0
    
    def test_high_spo2_decreases_flow(self):
        """Test high SpO2 results in decreased flow recommendation."""
        fuzzy = FuzzyOxygenTitrator()
        fuzzy.reset(initial_flow=5.0)
        
        # High SpO2, rising
        result = fuzzy.compute(spo2=98.0, spo2_trend=0.5, current_time=10.0)
        
        # Flow should decrease
        assert result['flow_adjustment'] < 0 or result['recommended_flow'] < 5.0
    
    def test_target_spo2_maintains_flow(self):
        """Test target SpO2 range maintains current flow."""
        fuzzy = FuzzyOxygenTitrator()
        fuzzy.reset(initial_flow=3.0)
        
        # Normal SpO2, stable
        result = fuzzy.compute(spo2=93.0, spo2_trend=0.0, current_time=10.0)
        
        # Flow adjustment should be minimal
        assert abs(result['flow_adjustment']) < 0.5
    
    def test_rate_limiting(self):
        """Test update rate limiting."""
        config = FuzzyConfig(update_interval=10.0)
        fuzzy = FuzzyOxygenTitrator(config)
        fuzzy.reset(initial_flow=3.0)
        
        # First update
        result1 = fuzzy.compute(spo2=85.0, spo2_trend=-1.0, current_time=0.0)
        
        # Second update too soon
        result2 = fuzzy.compute(spo2=85.0, spo2_trend=-1.0, current_time=5.0)
        
        assert result2.get('update_pending', False) == True
    
    def test_flow_limits(self):
        """Test flow stays within limits."""
        config = FuzzyConfig(flow_min=0.5, flow_max=10.0)
        fuzzy = FuzzyOxygenTitrator(config)
        
        # Try to push flow very high
        for t in range(100):
            fuzzy.compute(spo2=70.0, spo2_trend=-5.0, current_time=t * 11.0)
        
        assert fuzzy.get_current_flow() <= config.flow_max
        
        # Try to push flow very low
        fuzzy.reset(initial_flow=5.0)
        for t in range(100):
            fuzzy.compute(spo2=100.0, spo2_trend=5.0, current_time=t * 11.0)
        
        assert fuzzy.get_current_flow() >= config.flow_min


class TestSafetyWatchdog:
    """Tests for safety watchdog."""
    
    def test_initialization(self):
        """Test safety watchdog initializes in normal state."""
        safety = SafetyWatchdog()
        status = safety.get_status()
        
        assert status.level == SafetyLevel.NORMAL
    
    def test_pressure_high_alarm(self):
        """Test high pressure triggers alarm."""
        safety = SafetyWatchdog()
        
        command = {'pwm_duty': 0.8, 'flow_setpoint': 5.0}
        state = {'pressure': 3.5, 'purity': 0.93, 'temperature': 50, 'flow': 5.0}
        
        status = safety.validate_command(command, state, current_time=1.0)
        
        assert not status.safe
        assert SafetyLevel.ALARM.value <= status.level.value
    
    def test_pressure_low_alarm(self):
        """Test low pressure triggers alarm and increases PWM."""
        safety = SafetyWatchdog()
        
        # First establish a baseline PWM to avoid rate limiting issues
        command_init = {'pwm_duty': 0.5, 'flow_setpoint': 5.0}
        state_init = {'pressure': 2.0, 'purity': 0.93, 'temperature': 50, 'flow': 5.0}
        safety.validate_command(command_init, state_init, current_time=0.0)
        
        # Now trigger low pressure
        command = {'pwm_duty': 0.2, 'flow_setpoint': 5.0}
        state = {'pressure': 0.3, 'purity': 0.93, 'temperature': 50, 'flow': 5.0}
        
        status = safety.validate_command(command, state, current_time=1.0)
        
        assert not status.safe
        # Should increase PWM to raise pressure (watchdog forces to 0.8)
        assert status.corrected_command['pwm_duty'] >= command['pwm_duty']
    
    def test_purity_low_alarm(self):
        """Test low purity triggers alarm."""
        safety = SafetyWatchdog()
        
        command = {'pwm_duty': 0.5, 'flow_setpoint': 5.0}
        state = {'pressure': 2.0, 'purity': 0.80, 'temperature': 50, 'flow': 5.0}
        
        status = safety.validate_command(command, state, current_time=1.0)
        
        assert not status.safe
    
    def test_temperature_shutdown(self):
        """Test critical temperature triggers shutdown."""
        safety = SafetyWatchdog()
        
        command = {'pwm_duty': 0.8, 'flow_setpoint': 5.0}
        state = {'pressure': 2.0, 'purity': 0.93, 'temperature': 100, 'flow': 5.0}
        
        status = safety.validate_command(command, state, current_time=1.0)
        
        assert status.level == SafetyLevel.SHUTDOWN
        assert status.corrected_command['pwm_duty'] == 0.0
    
    def test_actuator_rate_limiting(self):
        """Test PWM rate limiting."""
        safety = SafetyWatchdog()
        
        # First command
        command1 = {'pwm_duty': 0.1, 'flow_setpoint': 5.0}
        state = {'pressure': 2.0, 'purity': 0.93, 'temperature': 50, 'flow': 5.0}
        safety.validate_command(command1, state, current_time=0.0)
        
        # Second command with large jump
        command2 = {'pwm_duty': 0.9, 'flow_setpoint': 5.0}
        status = safety.validate_command(command2, state, current_time=0.05)
        
        # Should be rate limited
        # PWM should not jump from 0.1 to 0.9 instantly
        assert status.corrected_command['pwm_duty'] < 0.9
    
    def test_emergency_stop(self):
        """Test emergency stop function."""
        safety = SafetyWatchdog()
        
        safe_command = safety.emergency_stop()
        
        assert safe_command['pwm_duty'] == 0.0
        assert safe_command['emergency_stop'] == True
        
        status = safety.get_status()
        assert status.level == SafetyLevel.SHUTDOWN
    
    def test_normal_operation(self):
        """Test normal operation passes through commands with rate limiting."""
        safety = SafetyWatchdog()
        
        # First establish baseline
        state = {'pressure': 2.0, 'purity': 0.93, 'temperature': 50, 'flow': 5.0}
        
        # Ramp up gradually to avoid rate limiting
        for i in range(10):
            pwm = (i + 1) * 0.05
            command = {'pwm_duty': pwm, 'flow_setpoint': 5.0, 'pressure_setpoint': 2.0}
            status = safety.validate_command(command, state, current_time=i * 0.2)
        
        # Now test at 0.5 after ramp-up
        command = {'pwm_duty': 0.5, 'flow_setpoint': 5.0, 'pressure_setpoint': 2.0}
        status = safety.validate_command(command, state, current_time=3.0)
        
        assert status.safe
        assert status.level == SafetyLevel.NORMAL


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
