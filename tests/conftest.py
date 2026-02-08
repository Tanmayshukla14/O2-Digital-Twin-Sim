"""
Pytest Configuration
====================

Shared fixtures and configuration for tests.
"""

import pytest
import numpy as np


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_sensor_state():
    """Sample sensor state for testing."""
    return {
        'pressure': 2.0,
        'purity': 0.93,
        'temperature': 50.0,
        'flow': 3.0,
        'motor_current': 5.0,
    }


@pytest.fixture
def sample_control_command():
    """Sample control command for testing."""
    return {
        'pwm_duty': 0.5,
        'flow_setpoint': 3.0,
        'pressure_setpoint': 2.0,
    }
