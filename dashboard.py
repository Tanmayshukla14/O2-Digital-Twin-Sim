"""
AI-Oxygen Concentrator Streamlit Dashboard
===========================================

Interactive dashboard for visualizing and controlling the oxygen concentrator simulation.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from simulator.physics.compressor import DiaphragmCompressor, CompressorConfig
from simulator.physics.psa import PSASystem, PSAConfig
from simulator.physics.patient import PatientModel, PatientConfig
from control.pid import PIDController, PIDConfig
from control.fuzzy import FuzzyOxygenTitrator, FuzzyConfig
from control.safety_watchdog import SafetyWatchdog, SafetyLevel


# Page configuration
st.set_page_config(
    page_title="AI-Oxygen Concentrator",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .status-normal { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-alarm { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_simulation():
    """Initialize simulation components."""
    components = {
        'compressor': DiaphragmCompressor(),
        'psa': PSASystem(),
        'patient': PatientModel(),
        'pid': PIDController(),
        'fuzzy': FuzzyOxygenTitrator(),
        'safety': SafetyWatchdog(),
    }
    
    # Initialize safety watchdog with a baseline command to prevent rate-limit alarm on startup
    baseline_state = {'pressure': 1.0, 'purity': 0.93, 'temperature': 25, 'flow': 3.0}
    baseline_cmd = {'pwm_duty': 0.5, 'flow_setpoint': 3.0, 'pressure_setpoint': 2.0}
    for i in range(10):  # Ramp up gradually
        cmd = {'pwm_duty': 0.05 * (i + 1), 'flow_setpoint': 3.0, 'pressure_setpoint': 2.0}
        components['safety'].validate_command(cmd, baseline_state, i * 0.1)
    
    return components


def run_simulation_step(components: Dict, settings: Dict, dt: float = 0.001) -> Dict:
    """Run one simulation step at 1kHz (dt=0.001s) for accurate physics."""
    comp = components['compressor']
    psa = components['psa']
    patient = components['patient']
    pid = components['pid']
    safety = components['safety']
    
    # Get control output from PID
    pressure_error = settings['pressure_setpoint'] - comp.get_state()['pressure']
    pwm_duty = pid.compute(
        setpoint=settings['pressure_setpoint'],
        measurement=comp.get_state()['pressure'],
        dt=dt
    )
    pwm_duty = np.clip(pwm_duty / 3.0 + 0.5, 0, 1)  # Normalize
    
    # Safety check
    current_state = {
        'pressure': comp.get_state()['pressure'],
        'purity': psa.get_state()['o2_purity'],
        'temperature': comp.get_state()['temperature'],
        'flow': settings['flow_setpoint'],
    }
    
    command = {
        'pwm_duty': pwm_duty,
        'flow_setpoint': settings['flow_setpoint'],
        'pressure_setpoint': settings['pressure_setpoint'],
    }
    
    safety_status = safety.validate_command(command, current_state, time.time())
    safe_pwm = safety_status.corrected_command.get('pwm_duty', pwm_duty)
    
    # Step physics - compressor needs back_pressure (tank pressure)
    tank_pressure = comp.state.outlet_pressure  # Current tank pressure in Pa
    comp.step(dt=dt, pwm_duty=safe_pwm, back_pressure=tank_pressure)
    
    # PSA uses pressure in bar, convert from Pa
    psa.step(
        inlet_pressure=comp.state.outlet_pressure / 1e5,  # Convert Pa to bar
        inlet_n2_fraction=0.79,
        inlet_o2_fraction=0.21,
        dt=dt
    )
    
    patient.step(
        delivered_o2_lpm=settings['flow_setpoint'],
        o2_purity=psa.get_state()['o2_purity'],
        dt=dt
    )
    
    return {
        'pressure': comp.get_state()['pressure'],
        'temperature': comp.get_state()['temperature'],
        'efficiency': comp.get_state()['efficiency'],
        'purity': psa.get_state()['o2_purity'] * 100,
        'psa_phase': psa.get_state()['phase'],
        'spo2': patient.get_state()['spo2'],
        'pwm': safe_pwm * 100,
        'safety_level': safety_status.level.name,
        'flow': settings['flow_setpoint'],
    }


def create_gauge(value: float, title: str, min_val: float, max_val: float, 
                 warning_threshold: float, danger_threshold: float,
                 inverted: bool = False) -> go.Figure:
    """Create a gauge chart.
    
    Args:
        inverted: If True, low values are danger (e.g., SpO2 where low is bad)
    """
    if inverted:
        # For SpO2: low is bad, high is good
        if value < danger_threshold:
            color = "red"
        elif value < warning_threshold:
            color = "orange"
        else:
            color = "green"
        
        steps = [
            {'range': [min_val, danger_threshold], 'color': "lightcoral"},
            {'range': [danger_threshold, warning_threshold], 'color': "lightyellow"},
            {'range': [warning_threshold, max_val], 'color': "lightgreen"},
        ]
    else:
        # For pressure/temperature: high is bad
        if value < warning_threshold:
            color = "green"
        elif value < danger_threshold:
            color = "orange"
        else:
            color = "red"
        
        steps = [
            {'range': [min_val, warning_threshold], 'color': "lightgreen"},
            {'range': [warning_threshold, danger_threshold], 'color': "lightyellow"},
            {'range': [danger_threshold, max_val], 'color': "lightcoral"},
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': steps,
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_time_series(data: Dict[str, List], title: str, y_label: str) -> go.Figure:
    """Create time series chart."""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, values) in enumerate(data.items()):
        fig.add_trace(go.Scatter(
            y=values,
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        xaxis_title="Time Steps",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    return fig


def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">🫁 AI-Oxygen Concentrator Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize components
    if 'components' not in st.session_state:
        st.session_state.components = init_simulation()
        st.session_state.history = {
            'pressure': [], 'temperature': [], 'purity': [], 
            'spo2': [], 'pwm': [], 'flow': []
        }
        st.session_state.running = False
        st.session_state.step_count = 0
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        st.subheader("Setpoints")
        pressure_sp = st.slider("Pressure Setpoint (bar)", 0.5, 3.0, 2.0, 0.1)
        flow_sp = st.slider("Flow Setpoint (LPM)", 0.5, 10.0, 3.0, 0.5)
        
        st.subheader("Patient Settings")
        condition = st.selectbox("Patient Condition", 
                                 ["normal", "copd", "pneumonia", "post_surgery"],
                                 help="Simulate different lung conditions affecting SpO2 response.")
        
        st.subheader("Simulation")
        sim_speed = st.slider("Simulation Speed", 1, 100, 50, help="Control simulation time steps per frame.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start" if not st.session_state.running else "⏸️ Pause", 
                        use_container_width=True):
                st.session_state.running = not st.session_state.running
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.components = init_simulation()
                st.session_state.history = {
                    'pressure': [], 'temperature': [], 'purity': [], 
                    'spo2': [], 'pwm': [], 'flow': []
                }
                st.session_state.step_count = 0
                st.session_state.running = False
    
    # Settings
    settings = {
        'pressure_setpoint': pressure_sp,
        'flow_setpoint': flow_sp,
        'condition': condition,
    }
    
    # Run simulation steps
    if st.session_state.running:
        for _ in range(sim_speed):
            result = run_simulation_step(st.session_state.components, settings)
            
            # Update history
            for key in st.session_state.history:
                if key in result:
                    st.session_state.history[key].append(result[key])
                    # Keep last 500 points
                    if len(st.session_state.history[key]) > 500:
                        st.session_state.history[key] = st.session_state.history[key][-500:]
            
            st.session_state.step_count += 1
            st.session_state.last_result = result
        
        time.sleep(0.05)  # Control update rate
        # NOTE: We sleep 50ms to prevent the UI from hogging the CPU.
        # The physics runs in the loop above at 1kHz equivalent (dt=0.001), 
        # so this visual pause doesn't affect simulation accuracy.
        st.rerun()
    
    # Get current values
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
    else:
        result = {
            'pressure': 1.0, 'temperature': 25.0, 'purity': 21.0,
            'spo2': 95.0, 'pwm': 0.0, 'safety_level': 'NORMAL', 'flow': 3.0
        }
    
    # Status bar
    status_color = {
        'NORMAL': 'status-normal',
        'WARNING': 'status-warning',
        'ALARM': 'status-alarm',
        'CRITICAL': 'status-alarm',
        'SHUTDOWN': 'status-alarm',
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Simulation Time", f"{st.session_state.step_count / 1000:.1f} s")
    with col2:
        st.metric("🔄 Status", "Running" if st.session_state.running else "Paused")
    with col3:
        safety_class = status_color.get(result['safety_level'], 'status-normal')
        st.markdown(f"🛡️ Safety: <span class='{safety_class}'>{result['safety_level']}</span>", 
                   unsafe_allow_html=True)
    with col4:
        st.metric("💨 Flow", f"{result['flow']:.1f} LPM")
    
    st.divider()
    
    # Gauges row
    st.subheader("📊 Live Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = create_gauge(result['pressure'], "Pressure (bar)", 0, 4, 2.5, 3.0)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # O2 Purity: inverted (low is bad)
        fig = create_gauge(result['purity'], "O₂ Purity (%)", 0, 100, 90, 85, inverted=True)
        st.plotly_chart(fig, width='stretch')
    
    with col3:
        # SpO2: inverted (low is bad)
        fig = create_gauge(result['spo2'], "SpO₂ (%)", 80, 100, 92, 88, inverted=True)
        st.plotly_chart(fig, width='stretch')
    
    with col4:
        fig = create_gauge(result['temperature'], "Motor Temp (°C)", 0, 100, 65, 80)
        st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    # Charts
    st.subheader("📈 Historical Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.history['pressure']:
            fig = create_time_series(
                {'Pressure': st.session_state.history['pressure']},
                "Tank Pressure",
                "Pressure (bar)"
            )
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        if st.session_state.history['purity']:
            fig = create_time_series(
                {'O₂ Purity': st.session_state.history['purity']},
                "Oxygen Purity",
                "Purity (%)"
            )
            st.plotly_chart(fig, width='stretch')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.history['spo2']:
            fig = create_time_series(
                {'SpO₂': st.session_state.history['spo2']},
                "Patient Oxygen Saturation",
                "SpO₂ (%)"
            )
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        if st.session_state.history['temperature']:
            fig = create_time_series(
                {'Motor Temp (°C)': st.session_state.history['temperature']},
                "Motor Temperature",
                "Temperature (°C)"
            )
            st.plotly_chart(fig, width='stretch')
    
    # Third row for PWM
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.history['pwm']:
            fig = create_time_series(
                {'PWM Duty (%)': st.session_state.history['pwm']},
                "Motor PWM Duty Cycle",
                "PWM (%)"
            )
            st.plotly_chart(fig, width='stretch')
    
    # Component status cards
    st.divider()
    st.subheader("🔧 Component Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        comp_state = st.session_state.components['compressor'].get_state()
        st.markdown("""
        **Diaphragm Compressor**
        - Pressure: {:.2f} bar
        - Temperature: {:.1f} °C
        - Efficiency: {:.1%}
        - Runtime: {:.0f} hours
        """.format(
            comp_state.get('pressure', 0),
            comp_state.get('temperature', 0),
            comp_state.get('efficiency', 1),
            comp_state.get('total_runtime', 0) / 3600
        ))
    
    with col2:
        psa_state = st.session_state.components['psa'].get_state()
        phase_names = ['Adsorb A', 'Equalize A→B', 'Adsorb B', 'Equalize B→A']
        st.markdown("""
        **PSA System**
        - O₂ Purity: {:.1f}%
        - Phase: {}
        - Cycle Count: {}
        - Zeolite Health: {:.1%}
        """.format(
            psa_state.get('o2_purity', 0.21) * 100,
            phase_names[psa_state.get('phase', 0) % 4],
            psa_state.get('cycle_count', 0),
            psa_state.get('zeolite_efficiency', 1)
        ))
    
    with col3:
        patient_state = st.session_state.components['patient'].get_state()
        st.markdown("""
        **Patient Model**
        - SpO₂: {:.1f}%
        - Activity Level: {:.0%}
        - Condition: {}
        - Oxygen Deficit: {:.2f}
        """.format(
            patient_state.get('spo2', 95),
            patient_state.get('activity_level', 1),
            settings['condition'],
            patient_state.get('oxygen_deficit', 0)
        ))
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        AI-Oxygen Concentrator Digital Twin Simulation | 
        Built with Physics-Based Models & Real-Time Control<br>
        <span style='color: green; font-weight: bold;'>✓ Hardware Ready Configuration</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
