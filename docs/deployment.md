# Deployment Guide: From Simulation to Hardware

This guide outlines the steps to deploy the AI-Oxygen Concentrator software to a physical hardware prototype.

## 1. Hardware Requirements

### Core Components
- **Microcontroller:** Raspberry Pi 4 (recommended) or Jetson Nano (for on-device AI training)
- **Compressor:** 12V/24V BLDC Diaphragm Compressor (Industrial Grade, >2.5 bar capable)
- **Valves:** 2x 3-way Solenoid Valves (12V/24V, High Flow Cv>0.05)
- **Sensors:**
    - Pressure Sensor (0-5 bar, I2C/Analog) - *Honeywell ABP series recommended*
    - Oxygen Purity Sensor (UART/I2C) - *Ultrasonic O2 sensor recommended*
    - Flow Sensor (Mass Flow Meter)

### Power System
- **Power Supply:** 24V 10A DC Power Supply
- **Motor Driver:** ESC (Electronic Speed Controller) for BLDC Motor (30A+)
- **Relay/MOSFET Module:** To drive solenoid valves

## 2. Wiring Diagram (Conceptual)
```
[Power Supply] --(24V)--> [Motor Driver] --(3-phase)--> [Compressor Motor]
                 |
                 +------> [Valve 1]
                 |
                 +------> [Valve 2]
                 |
                 +--(5V)-> [Raspberry Pi] --(USB/UART)--> [Sensors]
                                    |
                                    +--(PWM)--> [Motor Driver Signal]
                                    |
                                    +--(GPIO)--> [Valve Relays]
```

## 3. Software Installation on Hardware

1.  **OS Setup:** Install Raspberry Pi OS (64-bit Lite) or Ubuntu Server.
2.  **Dependencies:**
    ```bash
    sudo apt update && sudo apt install python3-pip python3-numpy
    pip3 install -r requirements.txt
    ```
3.  **Hardware Interface:**
    - The simulation uses `simulator.physics` modules. specific hardware drivers must replace these.
    - Create a `hardware/` directory and implement `CompressorInterface` and `SensorInterface` classes that match the simulator's API.

## 4. Calibration & Safety Checks

### Initial Power-Up
1.  **Dry Run:** Disconnect motor and valves. Run software to verify sensor readings are roughly atmospheric (1.0 bar, 21% O2).
2.  **Valve Check:** Manually toggle GPIOs to confirm valve clicking.
3.  **Leak Test:** pressurize system to 2 bar and check for leaks using soapy water.

### PID Tuning
1.  Start with low P, I, D gains.
2.  Set pressure target to 1.5 bar.
3.  Increase P until oscillation, then reduce by 50%.
4.  Increase I to remove steady-state error.
5.  Increase D to dampen overshoot.

## 5. Safety Protocols

**CRITICAL:** Ensure physical relief valves are installed!
- **Mechanical Relief Valve:** Set to 3.5 bar (Hard limit).
- **Watchdog Timer:** Enable hardware watchdog on the Pi to reboot if software hangs.
- **Thermal Cutoff:** Hardware thermal switch on the motor (85°C cutoff).

## 6. Going Live
1.  Run `dashboard.py` in "Hardware Mode" (requires code flag update).
2.  Monitor "Safety Level" constantly.
3.  Keep `Emergency Stop` button accessible.
