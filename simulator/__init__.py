"""
AI-Oxygen Concentrator Digital Twin Simulator
=============================================

Physics-based simulation environment for training and validating
AI-integrated oxygen concentrator control systems.
"""

__version__ = "1.0.0"
__author__ = "AI-Oxygen Research Team"

from .environment import OxygenConcentratorEnv
from .runner import SimulationRunner

__all__ = [
    "OxygenConcentratorEnv",
    "SimulationRunner",
]
