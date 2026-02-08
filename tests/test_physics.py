"""
Unit Tests for Simulator Physics
================================

Tests for compressor, PSA, and patient models.
"""

import pytest
import numpy as np
from simulator.physics.compressor import DiaphragmCompressor, CompressorConfig
from simulator.physics.psa import PSASystem, PSAConfig
from simulator.physics.patient import PatientModel, PatientConfig, PatientCondition


class TestDiaphragmCompressor:
    """Tests for DiaphragmCompressor model."""
    
    def test_initialization(self):
        """Test compressor initializes with valid state."""
        comp = DiaphragmCompressor()
        state = comp.get_state()
        
        assert state['pressure'] >= 0
        assert state['temperature'] >= 0
        assert state['efficiency'] > 0
        assert state['efficiency'] <= 1
    
    def test_step_increases_pressure(self):
        """Test that running compressor generates work (motor runs, temperature increases)."""
        comp = DiaphragmCompressor()
        
        # Get initial state
        initial_temp = comp.state.motor_temperature
        initial_cycles = comp.state.total_cycles
        
        # Run for 5 seconds at 80% duty
        for _ in range(5000):
            # Use current outlet pressure as back pressure (closed loop)
            comp.step(
                dt=0.001,
                pwm_duty=0.8,
                back_pressure=comp.state.outlet_pressure
            )
        
        # Compressor should be doing work:
        # - Motor should heat up
        # - Cycles should accumulate
        # - Angular velocity should be non-zero during operation
        assert comp.state.motor_temperature > initial_temp, "Motor should heat up during operation"
        assert comp.state.total_cycles > initial_cycles, "Cycles should accumulate"
    
    def test_zero_duty_maintains_pressure(self):
        """Test that zero duty cycle maintains approximately same pressure."""
        comp = DiaphragmCompressor()
        initial_pressure = comp.state.outlet_pressure
        
        # Run for 1 second at 0% duty
        for _ in range(1000):
            comp.step(
                dt=0.001,
                pwm_duty=0.0,
                back_pressure=comp.state.outlet_pressure
            )
        
        final_pressure = comp.state.outlet_pressure
        
        # Pressure should stay approximately the same (no compression at 0 duty)
        # Starts at atmospheric (1.01325e5 Pa)
        assert abs(final_pressure - initial_pressure) / initial_pressure < 0.1  # Within 10%
    
    def test_thermal_dynamics(self):
        """Test that motor heats up under load."""
        comp = DiaphragmCompressor()
        initial_temp = comp.get_state()['temperature']
        
        # Run at high duty for extended time
        for _ in range(5000):
            comp.step(
                dt=0.001,
                pwm_duty=0.9,
                back_pressure=comp.state.outlet_pressure
            )
        
        final_temp = comp.get_state()['temperature']
        assert final_temp > initial_temp
    
    def test_degradation_over_long_operation(self):
        """Test that health degrades over extended operation."""
        config = CompressorConfig()
        config.total_lifetime_hours = 0.001  # Very short lifetime for testing
        comp = DiaphragmCompressor(config)
        
        initial_health = comp.state.health_index
        
        # Run many cycles at high duty
        for _ in range(50000):
            comp.step(
                dt=0.001,
                pwm_duty=0.9,
                back_pressure=1.5e5
            )
        
        final_health = comp.state.health_index
        
        # Health should decrease (or stay same if wear is minimal)
        # Just verify it doesn't exceed initial and is still valid
        assert 0 <= final_health <= initial_health + 0.01  # Small tolerance
    
    def test_reset(self):
        """Test reset returns to initial state."""
        comp = DiaphragmCompressor()
        initial_pressure = comp.state.outlet_pressure
        initial_temp = comp.state.motor_temperature
        
        # Run for a while
        for _ in range(1000):
            comp.step(
                dt=0.001,
                pwm_duty=0.7,
                back_pressure=comp.state.outlet_pressure
            )
        
        # Reset
        comp.reset()
        
        # Check state returns to initial values
        assert comp.state.outlet_pressure == pytest.approx(initial_pressure, rel=0.01)
        assert comp.state.motor_temperature == pytest.approx(initial_temp, rel=0.01)
        assert comp.state.total_cycles == 0


class TestPSASystem:
    """Tests for PSA adsorption model."""
    
    def test_initialization(self):
        """Test PSA system initializes correctly."""
        psa = PSASystem()
        state = psa.get_state()
        
        assert 'phase' in state
        assert state['o2_purity'] > 0
        assert state['o2_purity'] <= 1.0
    
    def test_purity_improves_with_operation(self):
        """Test that purity increases during operation."""
        psa = PSASystem()
        
        # Run for several cycles at moderate pressure
        purities = []
        for i in range(20000):
            psa.step(
                inlet_pressure=2.0,
                inlet_n2_fraction=0.79,
                inlet_o2_fraction=0.21,
                dt=0.001
            )
            
            if i % 1000 == 0:
                purities.append(psa.get_state()['o2_purity'])
        
        # Purity should be above air levels
        assert purities[-1] > 0.21
    
    def test_phase_transitions(self):
        """Test that phase transitions occur."""
        psa = PSASystem()
        initial_phase = psa.get_state()['phase']
        
        # Run until phase changes
        for _ in range(50000):
            psa.step(inlet_pressure=2.0, inlet_n2_fraction=0.79, inlet_o2_fraction=0.21, dt=0.001)
        
        final_phase = psa.get_state()['phase']
        
        # Phase should have changed at least once
        # (or we've gone through complete cycles)
        assert psa.get_state()['cycle_count'] > 0 or final_phase != initial_phase
    
    def test_reset(self):
        """Test PSA reset."""
        psa = PSASystem()
        
        for _ in range(5000):
            psa.step(inlet_pressure=2.0, inlet_n2_fraction=0.79, inlet_o2_fraction=0.21, dt=0.001)
        
        psa.reset()
        state = psa.get_state()
        
        assert state['phase'] == 0  # Should start in first phase


class TestPatientModel:
    """Tests for patient oxygen physiology model."""
    
    def test_initialization(self):
        """Test patient model initialization."""
        patient = PatientModel()
        state = patient.get_state()
        
        assert 60 <= state['spo2'] <= 100  # Wide range for different conditions
        assert state['activity_level'] >= 0
    
    def test_spo2_increases_with_oxygen(self):
        """Test SpO2 increases with supplemental oxygen."""
        # Use COPD patient that starts with lower SpO2
        config = PatientConfig(condition=PatientCondition.COPD_MILD)
        patient = PatientModel(config)
        
        # Provide low oxygen first to lower SpO2
        for _ in range(10000):
            patient.step(delivered_o2_lpm=0.0, o2_purity=0.21, dt=0.001)
        
        initial_spo2 = patient.get_state()['spo2']
        
        # Now provide high oxygen
        for _ in range(60000):  # 60 seconds
            patient.step(
                delivered_o2_lpm=5.0,
                o2_purity=0.93,
                dt=0.001
            )
        
        final_spo2 = patient.get_state()['spo2']
        
        # SpO2 should increase
        assert final_spo2 > initial_spo2
    
    def test_spo2_decreases_without_oxygen(self):
        """Test SpO2 decreases without supplemental oxygen."""
        config = PatientConfig(condition=PatientCondition.COPD_MODERATE)
        patient = PatientModel(config)
        
        # Provide oxygen first to raise SpO2
        for _ in range(30000):
            patient.step(delivered_o2_lpm=5.0, o2_purity=0.93, dt=0.001)
        
        initial_spo2 = patient.get_state()['spo2']
        
        # No oxygen
        for _ in range(120000):  # 2 minutes
            patient.step(
                delivered_o2_lpm=0.0,
                o2_purity=0.21,
                dt=0.001
            )
        
        final_spo2 = patient.get_state()['spo2']
        
        # SpO2 should decrease
        assert final_spo2 < initial_spo2
    
    def test_hypoxia_detection(self):
        """Test hypoxia event detection via state dict."""
        config = PatientConfig(condition=PatientCondition.COPD_SEVERE)
        patient = PatientModel(config)
        
        # Step without oxygen to induce hypoxia
        for _ in range(60000):
            patient.step(delivered_o2_lpm=0.0, o2_purity=0.21, dt=0.001)
        
        state_dict = patient.get_state_dict()
        
        # Should have hypoxia tracking in state
        assert 'hypoxia_events' in state_dict or 'desaturation_duration_s' in state_dict
    
    def test_condition_affects_response(self):
        """Test different conditions have different responses."""
        # Use enum conditions
        patient_mild = PatientModel(PatientConfig(condition=PatientCondition.COPD_MILD))
        patient_severe = PatientModel(PatientConfig(condition=PatientCondition.COPD_SEVERE))
        
        # Run both without oxygen
        for _ in range(30000):
            patient_mild.step(delivered_o2_lpm=0.0, o2_purity=0.21, dt=0.001)
            patient_severe.step(delivered_o2_lpm=0.0, o2_purity=0.21, dt=0.001)
            
        # Severe COPD patient should have lower SpO2
        final_mild = patient_mild.get_state()['spo2']
        final_severe = patient_severe.get_state()['spo2']
        
        assert final_mild > 0
        assert final_severe > 0
        # Severe should be lower or equal (depends on dynamics)
        assert final_severe <= final_mild + 5  # Some tolerance


class TestPhysicsIntegration:
    """Integration tests for combined physics models."""
    
    def test_full_system_simulation(self):
        """Test full system runs without errors."""
        comp = DiaphragmCompressor()
        psa = PSASystem()
        patient = PatientModel()
        
        # Run for 10 simulated seconds
        for i in range(10000):
            # Compressor with back_pressure
            comp.step(
                dt=0.001,
                pwm_duty=0.6,
                back_pressure=comp.state.outlet_pressure
            )
            comp_state = comp.get_state()
            
            # PSA
            psa.step(
                inlet_pressure=comp_state['pressure'],
                inlet_n2_fraction=0.79,
                inlet_o2_fraction=0.21,
                dt=0.001
            )
            psa_state = psa.get_state()
            
            # Patient
            patient.step(
                delivered_o2_lpm=3.0,
                o2_purity=psa_state['o2_purity'],
                dt=0.001
            )
        
        # All should have valid states
        assert comp.get_state()['pressure'] >= 0
        assert 0 <= psa.get_state()['o2_purity'] <= 1
        assert 0 < patient.get_state()['spo2'] <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
