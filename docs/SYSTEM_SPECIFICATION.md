# System Specification: AI-Integrated Diaphragm Oxygen Concentrator

**Document Version**: 1.0  
**Classification**: Engineering Specification  

---

## 1. SYSTEM OVERVIEW

### 1.1 Purpose

This system provides autonomous oxygen concentration with AI-optimized efficiency for medical applications. The design integrates a digital twin simulator for offline AI training with real-time embedded control.

### 1.2 Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| FR-01 | Oxygen purity output | ≥ 87% (nominal 93%) |
| FR-02 | Flow rate range | 0.5 - 10 LPM |
| FR-03 | Control loop frequency | 1 kHz |
| FR-04 | AI inference frequency | 10 Hz |
| FR-05 | Patient SpO2 maintenance | 88-100% |
| FR-06 | Power optimization | Minimize energy/L O2 |

### 1.3 Non-Functional Requirements

| ID | Requirement | Specification |
|----|-------------|---------------|
| NFR-01 | Safety response time | < 10 ms |
| NFR-02 | MCU memory footprint | < 256 KB RAM |
| NFR-03 | AI model size | < 50 KB (TFLite) |
| NFR-04 | MTBF | > 10,000 hours |

---

## 2. ARCHITECTURE

### 2.1 System Block Diagram

```
                                    ┌──────────────────────────────────┐
                                    │         USER INTERFACE           │
                                    │   (Display / Alarms / Config)    │
                                    └──────────────────────────────────┘
                                                     │
    ┌────────────────────────────────────────────────┼────────────────────────────────────────────────┐
    │                                    EMBEDDED CONTROLLER                                          │
    │  ┌─────────────────────────────────────────────┴─────────────────────────────────────────────┐ │
    │  │                              SAFETY SUPERVISOR (Highest Priority)                          │ │
    │  │   • Hard pressure limits [0.5, 3.0] bar    • Thermal protection < 85°C                    │ │
    │  │   • Purity minimum ≥ 87%                   • Actuator rate limiting                       │ │
    │  │   • Watchdog timer                         • Emergency shutdown logic                      │ │
    │  └───────────────────────────────────────────────────────────────────────────────────────────┘ │
    │                                                │                                                │
    │  ┌──────────────────┐    ┌──────────────────┐ │ ┌──────────────────┐    ┌──────────────────┐  │
    │  │   SENSOR TASK    │    │  CONTROL TASK    │◄┼─┤   AI TASK        │    │   COMM TASK      │  │
    │  │     (1 kHz)      │───▶│    (1 kHz)       │ │ │   (10 Hz)        │    │   (async)        │  │
    │  │                  │    │                  │ │ │                  │    │                  │  │
    │  │ • ADC sampling   │    │ • PID control    │ │ │ • TFLite infer   │    │ • UART/BLE       │  │
    │  │ • Filtering      │    │ • Fuzzy blend    │ │ │ • DQN action     │    │ • Data logging   │  │
    │  │ • Calibration    │    │ • PWM output     │ │ │ • LSTM health    │    │                  │  │
    │  └──────────────────┘    └──────────────────┘ │ └──────────────────┘    └──────────────────┘  │
    │           │                       │           │          │                       │            │
    └───────────┼───────────────────────┼───────────┼──────────┼───────────────────────┼────────────┘
                │                       │           │          │                       │
    ┌───────────▼───────────────────────▼───────────┴──────────▼───────────────────────▼────────────┐
    │                                      HARDWARE LAYER                                            │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
    │  │  Pressure  │  │  O2 Sensor │  │   Temp     │  │   Motor    │  │  Solenoid  │              │
    │  │  (×3)      │  │  (Zirconia)│  │   (×2)     │  │   Driver   │  │  Valves    │              │
    │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘              │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
                │                       │                       │
    ┌───────────▼───────────────────────▼───────────────────────▼──────────────────────────────────┐
    │                                   PNEUMATIC SYSTEM                                            │
    │  ┌─────────────────┐    ┌─────────────────────────────────────┐    ┌─────────────────────┐   │
    │  │    DIAPHRAGM    │───▶│           PSA SYSTEM                │───▶│   PRODUCT TANK     │   │
    │  │   COMPRESSOR    │    │  ┌─────────┐       ┌─────────┐      │    │   + Flow Control   │   │
    │  │                 │    │  │  Bed A  │◄─────▶│  Bed B  │      │    │                    │   │
    │  │  • Motor driven │    │  │(Zeolite)│       │(Zeolite)│      │    │  • Buffer volume   │   │
    │  │  • 0.5-3.0 bar  │    │  └─────────┘       └─────────┘      │    │  • Flow valve      │   │
    │  │  • Oil-free     │    │      Cycle: Adsorb ←→ Desorb       │    │  • To patient      │   │
    │  └─────────────────┘    └─────────────────────────────────────┘    └─────────────────────┘   │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Sensors ──▶ Filter ──▶ State Vector ──▶ AI Optimizer ──▶ Setpoint Adjust
                              │                               │
                              ▼                               ▼
                        PID Controller ◄──────────────── Fuzzy Blend
                              │
                              ▼
                      Safety Validator ──▶ Actuator Commands
```

---

## 3. MATHEMATICAL MODELS

### 3.1 Diaphragm Compressor Dynamics

**Pressure-Volume Relationship (Polytropic Process):**

$$P_1 V_1^n = P_2 V_2^n$$

Where:
- $P$ = Pressure [Pa]
- $V$ = Volume [m³]  
- $n$ = Polytropic index (1.0 ≤ n ≤ 1.4)

**Diaphragm Displacement Model:**

$$x(t) = \frac{S}{2}(1 - \cos(\omega t))$$

Where:
- $S$ = Stroke length [m]
- $\omega$ = Angular velocity [rad/s]

**Instantaneous Pressure:**

$$P(t) = P_{atm} \left(\frac{V_0}{V_0 - A_d \cdot x(t)}\right)^n$$

Where:
- $V_0$ = Dead volume [m³]
- $A_d$ = Diaphragm area [m²]

**Pressure Dynamics (State Equation):**

$$\frac{dP_{tank}}{dt} = \frac{\gamma}{V_{tank}}(Q_{in} - Q_{out})$$

Where:
- $\gamma$ = Specific heat ratio (1.4 for air)
- $Q_{in}$ = Inlet flow [m³/s]
- $Q_{out}$ = Outlet flow [m³/s]

**Motor Thermal Model:**

$$\frac{dT_{motor}}{dt} = \frac{1}{C_{th}}(P_{loss} - \frac{T_{motor} - T_{amb}}{R_{th}})$$

Where:
- $C_{th}$ = Thermal capacitance [J/K]
- $R_{th}$ = Thermal resistance [K/W]
- $P_{loss} = I^2 R$ = Resistive losses [W]

### 3.2 PSA Adsorption Kinetics

**Langmuir Isotherm:**

$$q^* = \frac{q_m \cdot b \cdot P}{1 + b \cdot P}$$

Where:
- $q^*$ = Equilibrium loading [mol/kg]
- $q_m$ = Maximum loading capacity [mol/kg]
- $b$ = Langmuir constant [1/Pa]
- $P$ = Partial pressure [Pa]

**Linear Driving Force (LDF) Model:**

$$\frac{dq}{dt} = k_{LDF}(q^* - q)$$

Where:
- $k_{LDF}$ = Mass transfer coefficient [1/s]
- Typical values: 0.01-0.1 s⁻¹

**Nitrogen/Oxygen Separation:**

$$\alpha_{N_2/O_2} = \frac{q_{N_2}/P_{N_2}}{q_{O_2}/P_{O_2}} \approx 2.5-3.0$$

**Oxygen Purity Evolution:**

$$\frac{dy_{O_2}}{dt} = \frac{1}{\tau_{PSA}}(y_{O_2}^{ss} - y_{O_2}) + \epsilon_{humidity}$$

Where:
- $\tau_{PSA}$ = PSA time constant (~10-30s)
- $y_{O_2}^{ss}$ = Steady-state purity (f(pressure, cycle time))
- $\epsilon_{humidity}$ = Humidity degradation factor

**Humidity Effect on Zeolite:**

$$\eta_{zeolite} = \eta_0 \cdot e^{-k_h \cdot RH}$$

Where:
- $\eta_0$ = Dry efficiency
- $k_h$ = Humidity sensitivity (~0.01-0.02 per %RH)
- $RH$ = Relative humidity [%]

### 3.3 Patient Oxygen Model

**SpO2 Response (First-Order Lag):**

$$\frac{dSpO_2}{dt} = \frac{1}{\tau_{SpO_2}}(SpO_{2,target} - SpO_2)$$

Where:
- $\tau_{SpO_2}$ = Time constant (30-60s typical)
- $SpO_{2,target}$ = f(FiO2, respiratory function)

**Oxygen Delivery Equation:**

$$DO_2 = Q_c \times CaO_2 = Q_c \times (1.34 \times Hb \times SpO_2 + 0.003 \times PaO_2)$$

**Metabolic Oxygen Demand:**

$$VO_2 = VO_{2,basal} \times (1 + k_{activity} \times Activity + k_{temp} \times \Delta T)$$

### 3.4 Mechanical Degradation

**Wear Model (Archard Equation):**

$$\frac{dW}{dt} = K \cdot \frac{F \cdot v}{H}$$

Where:
- $W$ = Wear volume [m³]
- $K$ = Wear coefficient
- $F$ = Normal force [N]
- $v$ = Sliding velocity [m/s]
- $H$ = Hardness [Pa]

**Health Index:**

$$H_{idx}(t) = 1 - \frac{\int_0^t degradation(\tau)d\tau}{Life_{total}}$$

### 3.5 Energy Model

**Compressor Power:**

$$P_{comp} = \frac{\dot{m} \cdot c_p \cdot T_1}{\eta_{isen}} \left[\left(\frac{P_2}{P_1}\right)^{\frac{\gamma-1}{\gamma}} - 1\right]$$

**Energy Efficiency Metric:**

$$\eta_{system} = \frac{Q_{O_2} \cdot \Delta P_{O_2}}{P_{electrical} \cdot t}$$

---

## 4. STATE DEFINITIONS

### 4.1 System State Vector (for AI)

```python
state = {
    # Pressures [bar]
    'P_tank': float,          # Product tank pressure
    'P_bed_a': float,         # PSA bed A pressure
    'P_bed_b': float,         # PSA bed B pressure
    
    # Concentrations [fraction]
    'y_O2': float,            # Oxygen purity (0-1)
    'y_O2_rate': float,       # Purity rate of change
    
    # Flow [LPM]
    'Q_product': float,       # Product flow rate
    'Q_demand': float,        # Patient demand
    
    # Thermal [°C]
    'T_motor': float,         # Motor temperature
    'T_ambient': float,       # Ambient temperature
    
    # Patient [%]
    'SpO2': float,            # Patient oxygen saturation
    'SpO2_trend': float,      # SpO2 trend (derivative)
    
    # Health [0-1]
    'H_compressor': float,    # Compressor health index
    'H_zeolite': float,       # Zeolite health index
    
    # Energy [W, Wh]
    'P_current': float,       # Instantaneous power
    'E_cumulative': float,    # Cumulative energy
    
    # Timing [s]
    't_cycle': float,         # Current PSA cycle time
    'phase': int              # PSA phase (0=adsorb, 1=desorb)
}
```

### 4.2 Action Space (DQN)

```python
actions = {
    0: 'DECREASE_PRESSURE_LARGE',   # -0.2 bar setpoint
    1: 'DECREASE_PRESSURE_SMALL',   # -0.05 bar setpoint
    2: 'MAINTAIN',                   # No change
    3: 'INCREASE_PRESSURE_SMALL',   # +0.05 bar setpoint
    4: 'INCREASE_PRESSURE_LARGE',   # +0.2 bar setpoint
    5: 'DECREASE_CYCLE_TIME',       # -1s PSA cycle
    6: 'INCREASE_CYCLE_TIME',       # +1s PSA cycle
    7: 'DECREASE_FLOW',             # -0.5 LPM
    8: 'INCREASE_FLOW'              # +0.5 LPM
}
```

### 4.3 Reward Function

```python
def compute_reward(state, action, next_state):
    # Primary: Patient safety (critical)
    r_spo2 = -100.0 if next_state['SpO2'] < 88 else 0
    r_spo2 += -10.0 * max(0, 94 - next_state['SpO2'])  # Penalty below target
    
    # Secondary: Oxygen purity (important)
    r_purity = -50.0 if next_state['y_O2'] < 0.87 else 0
    r_purity += 5.0 * min(next_state['y_O2'], 0.96)  # Reward high purity
    
    # Tertiary: Energy efficiency (optimization)
    r_energy = -0.1 * next_state['P_current']  # Penalize power consumption
    
    # Stability: Penalize oscillations
    r_stability = -1.0 * abs(next_state['y_O2_rate'])
    
    # Health: Reward conservative operation
    r_health = 2.0 * next_state['H_compressor']
    
    # Safety constraint violation
    r_safety = -1000.0 if safety_violated(next_state) else 0
    
    return r_spo2 + r_purity + r_energy + r_stability + r_health + r_safety
```

---

## 5. SAFETY INVARIANTS

### 5.1 Formal Safety Constraints

```
INVARIANT pressure_bounds:
    ∀t: 0.5 bar ≤ P_tank(t) ≤ 3.0 bar

INVARIANT purity_minimum:
    ∀t: y_O2(t) ≥ 0.87 OR alarm_active(t)

INVARIANT thermal_protection:
    ∀t: T_motor(t) < 85°C

INVARIANT actuator_limits:
    ∀t: |dPWM/dt| ≤ 10% per 100ms

INVARIANT flow_bounds:
    ∀t: 0.0 LPM ≤ Q_product(t) ≤ 10.0 LPM

INVARIANT ai_override:
    ∀t: safety_check(ai_output(t)) = PASS 
        BEFORE actuator_apply(ai_output(t))
```

### 5.2 Failure Mode Analysis

| Failure Mode | Detection | Response | Recovery |
|--------------|-----------|----------|----------|
| Over-pressure | P > 3.0 bar | Vent valve open | Auto-reset at P < 2.5 |
| Under-pressure | P < 0.5 bar | Increase motor speed | Auto-reset at P > 0.7 |
| Low purity | y_O2 < 87% | Extend cycle, alarm | Auto-reset at y_O2 > 90% |
| Over-temp | T > 85°C | Reduce duty, alarm | Auto-reset at T < 70°C |
| Sensor fault | Value out of range | Use backup/default | Manual intervention |
| AI timeout | No output > 200ms | Use last valid PID | Auto-resume on recovery |

---

## 6. INTERFACE DEFINITIONS

### 6.1 Simulator API

```python
class OxygenConcentratorEnv:
    """Gym-compatible simulation environment."""
    
    def __init__(self, config: dict) -> None: ...
    def reset(self) -> np.ndarray: ...
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]: ...
    def render(self, mode: str = 'human') -> None: ...
    def get_state_dict(self) -> dict: ...
    def set_patient_profile(self, profile: PatientProfile) -> None: ...
```

### 6.2 Controller Interface

```python
class Controller(ABC):
    """Abstract base class for all controllers."""
    
    @abstractmethod
    def compute(self, state: dict, setpoint: dict) -> dict: ...
    
    @abstractmethod  
    def reset(self) -> None: ...
    
    @abstractmethod
    def get_telemetry(self) -> dict: ...
```

### 6.3 Safety Watchdog Interface

```python
class SafetyWatchdog:
    """Deterministic safety layer - cannot be bypassed."""
    
    def validate(self, command: dict) -> Tuple[dict, List[str]]: ...
    def check_state(self, state: dict) -> SafetyStatus: ...
    def emergency_stop(self) -> None: ...
    def get_active_constraints(self) -> List[str]: ...
```

### 6.4 Firmware Task Interfaces (C)

```c
// Sensor Task
void sensor_task(void *pvParameters);
SensorData_t sensor_get_latest(void);
bool sensor_is_valid(SensorType_t type);

// Control Task
void control_task(void *pvParameters);
void control_set_setpoint(Setpoint_t sp);
ControlOutput_t control_get_output(void);

// AI Task
void ai_task(void *pvParameters);
AIRecommendation_t ai_get_recommendation(void);
float ai_get_health_score(void);

// Safety Task
void safety_task(void *pvParameters);
SafetyStatus_t safety_get_status(void);
void safety_emergency_stop(void);
```

---

## 7. TIMING REQUIREMENTS

| Task | Period | Deadline | Priority |
|------|--------|----------|----------|
| Safety Watchdog | 1 ms | 1 ms | Highest (7) |
| Sensor Acquisition | 1 ms | 1 ms | High (6) |
| PID Control | 1 ms | 1 ms | High (5) |
| AI Inference | 100 ms | 100 ms | Medium (3) |
| Data Logging | 1000 ms | 5000 ms | Low (2) |
| Communication | Async | 100 ms | Low (1) |

---

## 8. MEMORY BUDGET (ESP32)

| Component | RAM | Flash |
|-----------|-----|-------|
| FreeRTOS Kernel | 20 KB | 50 KB |
| Sensor Buffers | 8 KB | - |
| Control State | 4 KB | - |
| AI Model (TFLite) | 30 KB | 50 KB |
| Inference Buffer | 16 KB | - |
| Data Logging | 32 KB | - |
| Stack Space | 24 KB | - |
| **Total** | **134 KB** | **100 KB** |
| **Available** | 320 KB | 4 MB |

---

## 9. CALIBRATION PARAMETERS

```yaml
compressor:
  stroke_length_m: 0.015
  diaphragm_area_m2: 0.002
  dead_volume_m3: 0.00005
  polytropic_n: 1.3
  motor_resistance_ohm: 2.5
  thermal_capacitance_JK: 150
  thermal_resistance_KW: 0.5

psa:
  bed_mass_kg: 0.5
  zeolite_qm_mol_kg: 4.5
  langmuir_b_1_Pa: 0.00001
  ldf_coefficient_1_s: 0.05
  cycle_time_nominal_s: 20
  humidity_sensitivity: 0.015

patient:
  spo2_time_constant_s: 45
  basal_vo2_ml_min: 250
  target_spo2_percent: 94

safety:
  pressure_min_bar: 0.5
  pressure_max_bar: 3.0
  purity_min_fraction: 0.87
  temp_max_celsius: 85
  flow_max_lpm: 10.0
  actuator_rate_limit_pct_s: 100
```
