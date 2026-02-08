# O2-Digital-Twin-Sim: Medical-Grade Physics & Hybrid Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Hardware Ready](https://img.shields.io/badge/Status-Hardware%20Ready-green.svg)](docs/deployment.md)
[![Security: Audited](https://img.shields.io/badge/Security-Audited-blue.svg)](docs/walkthrough.md)

## Overview

**O2-Digital-Twin-Sim** is a high-fidelity physics engine and hybrid control system for medical oxygen concentrators. Unlike standard end-to-end RL models, this project focuses on **Deterministic Safety** and **Digital Twin Accuracy**.

It serves as a "Ground Truth" verification tool for validating AI strategies before deployment.

### 🆚 How this differs from Pure RL (PPO) Models
| Feature | **O2-Digital-Twin-Sim** (This Repo) | **Standard RL / PPO Models** |
| :--- | :--- | :--- |
| **Control Logic** | **Hybrid:** PID + Fuzzy Logic + DQN (Optimizer) | End-to-End Neural Network Policy |
| **Physics Fidelity** | **High:** Simulates porous media flow, thermodynamics, diaphragm deflection | Low/Medium: Often simplified fluid dynamics |
| **Safety** | **Deterministic Watchdog** (Hard-coded physics constraints) | Probabilistic (Reward-based safety) |
| **Primary Goal** | **Hardware Reliability & Validation** | Maximizing Reward Functions |

---

### Key Features

- **Medical-Grade Physics**: accurate simulation of Li-LSX zeolite and industrial diaphragm compressors (3.0 bar / 99% purity).
- **Digital Twin**: Real-time simulation of pneumatic dynamics, thermodynamics, and patient physiology.
- **AI-Integrated Control**: Hybrid PID + Fuzzy Logic controller for precise pressure and oxygen titration.
- **Hardware Ready**: Deployment guides included for Raspberry Pi + Industrial Compressor builds.
- **Safety First**: Deterministic `SafetyWatchdog` module with hard real-time constraints and fuzz-tested robustness.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ai-oxygen-concentrator.git
cd ai-oxygen-concentrator

# Create virtual environment (Python 3.11+)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard

Launching the interactive digital twin:

```bash
streamlit run dashboard.py
```

Access the dashboard at `http://localhost:8501`.

### 3. Run Tests

Verify system integrity and physics accuracy:

```bash
python -m pytest tests/ -v
```

## 📚 Documentation

- **[Hardware Deployment Guide](docs/deployment.md)**: Step-by-step instructions for building the physical device.
- **[System Walkthrough & Validation](docs/walkthrough.md)**: Detailed report on system performance, upgrades, and security audit results.

## Architecture

The system follows a modular architecture designed for safety and scalability:

```
┌──────────────────────┐      ┌──────────────────────┐
│  Streamlit Dashboard │◄────►│   Control System     │
│   (User Interface)   │      │ (PID, Fuzzy, Safety) │
└──────────┬───────────┘      └──────────┬───────────┘
           │                             │
           ▼                             ▼
┌────────────────────────────────────────────────────┐
│                  PHYSICS SIMULATION                │
│ ┌────────────┐   ┌────────────┐   ┌────────────┐   │
│ │ Compressor │──►│ PSA System │──►│ Patient    │   │
│ └────────────┘   └────────────┘   └────────────┘   │
└────────────────────────────────────────────────────┘
```

## Safety Invariants

The `SafetyWatchdog` enforces the following hard limits:

| Parameter | Min | Max | Action |
|-----------|-----|-----|--------|
| **Pressure** | 0.5 bar | 3.0 bar | Alarm / PWM Reduction |
| **O₂ Purity** | 87% | 100% | Cycle Adjustment |
| **Temperature**| 0°C | 85°C | Thermal Shutdown |

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**Disclaimer:** This software is for research and prototyping purposes. While designed with medical-grade specifications, strict hardware validation and regulatory approval are required for clinical use.
