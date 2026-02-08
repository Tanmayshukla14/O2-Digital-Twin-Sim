/*
 * Control Task
 * ============
 * 
 * Real-time PID control loop at 1kHz.
 * - Cascade pressure/purity control
 * - PSA cycle management
 * - AI recommendation integration
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "config.h"
#include "types.h"

/* External references */
extern QueueHandle_t g_sensor_queue;
extern QueueHandle_t g_control_queue;
extern SystemState_t system_get_state(void);
extern void system_update_outputs(const ControlOutput_t *outputs);

/* PID controllers */
static PIDState_t pid_pressure;
static PIDState_t pid_flow;

/* PSA state machine */
static PSAState_t psa_state;

/**
 * Initialize PID controller
 */
static void pid_init(PIDState_t *pid, fixed16_t kp, fixed16_t ki, fixed16_t kd)
{
    memset(pid, 0, sizeof(PIDState_t));
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
}

/**
 * PID compute step (fixed-point)
 */
static fixed16_t pid_compute(PIDState_t *pid, fixed16_t setpoint, fixed16_t measurement)
{
    /* Error */
    fixed16_t error = setpoint - measurement;
    
    /* Proportional */
    fixed16_t p_term = (pid->kp * error) >> 16;
    
    /* Integral with anti-windup */
    pid->integral += (pid->ki * error) >> 16;
    
    if (pid->integral > PID_INTEGRAL_MAX) {
        pid->integral = PID_INTEGRAL_MAX;
        pid->saturated = true;
    } else if (pid->integral < PID_INTEGRAL_MIN) {
        pid->integral = PID_INTEGRAL_MIN;
        pid->saturated = true;
    } else {
        pid->saturated = false;
    }
    
    fixed16_t i_term = pid->integral;
    
    /* Derivative (on measurement to avoid derivative kick) */
    fixed16_t derivative = measurement - pid->prev_error;
    pid->prev_error = measurement;
    fixed16_t d_term = (pid->kd * (-derivative)) >> 16;
    
    /* Output */
    pid->output = p_term + i_term + d_term;
    
    return pid->output;
}

/**
 * Reset PID controller
 */
static void pid_reset(PIDState_t *pid)
{
    pid->integral = 0;
    pid->prev_error = 0;
    pid->output = 0;
    pid->saturated = false;
}

/**
 * Update PSA cycle state machine
 */
static void psa_update(uint32_t current_time_ms, uint16_t cycle_time_ms)
{
    uint32_t phase_duration = current_time_ms - psa_state.phase_start_time_ms;
    
    switch (psa_state.current_phase) {
        case PSA_PHASE_ADSORB_A:
            if (phase_duration >= cycle_time_ms) {
                psa_state.current_phase = PSA_PHASE_EQUALIZE_AB;
                psa_state.phase_start_time_ms = current_time_ms;
            }
            break;
            
        case PSA_PHASE_EQUALIZE_AB:
            if (phase_duration >= PSA_EQUALIZATION_TIME_MS) {
                psa_state.current_phase = PSA_PHASE_ADSORB_B;
                psa_state.phase_start_time_ms = current_time_ms;
                psa_state.cycle_count++;
            }
            break;
            
        case PSA_PHASE_ADSORB_B:
            if (phase_duration >= cycle_time_ms) {
                psa_state.current_phase = PSA_PHASE_EQUALIZE_BA;
                psa_state.phase_start_time_ms = current_time_ms;
            }
            break;
            
        case PSA_PHASE_EQUALIZE_BA:
            if (phase_duration >= PSA_EQUALIZATION_TIME_MS) {
                psa_state.current_phase = PSA_PHASE_ADSORB_A;
                psa_state.phase_start_time_ms = current_time_ms;
                psa_state.cycle_count++;
            }
            break;
    }
}

/**
 * Compute valve states based on PSA phase
 */
static void psa_get_valve_states(uint16_t *valve_a, uint16_t *valve_b)
{
    switch (psa_state.current_phase) {
        case PSA_PHASE_ADSORB_A:
            *valve_a = PWM_OUTPUT_MAX;  /* A open (adsorbing) */
            *valve_b = 0;               /* B closed (desorbing) */
            break;
            
        case PSA_PHASE_EQUALIZE_AB:
        case PSA_PHASE_EQUALIZE_BA:
            *valve_a = PWM_OUTPUT_MAX / 2;  /* Both partially open */
            *valve_b = PWM_OUTPUT_MAX / 2;
            break;
            
        case PSA_PHASE_ADSORB_B:
            *valve_a = 0;               /* A closed (desorbing) */
            *valve_b = PWM_OUTPUT_MAX;  /* B open (adsorbing) */
            break;
    }
}

/**
 * Control task main loop
 */
void control_task(void *pvParameters)
{
    (void)pvParameters;
    
    /* Initialize PIDs */
    pid_init(&pid_pressure, PID_KP_DEFAULT, PID_KI_DEFAULT, PID_KD_DEFAULT);
    pid_init(&pid_flow, FLOAT_TO_FIXED16(1.0), FLOAT_TO_FIXED16(0.2), FLOAT_TO_FIXED16(0.05));
    
    /* Initialize PSA */
    memset(&psa_state, 0, sizeof(PSAState_t));
    psa_state.current_phase = PSA_PHASE_ADSORB_A;
    psa_state.phase_start_time_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
    
    TickType_t last_wake = xTaskGetTickCount();
    SensorData_t sensors;
    ControlOutput_t outputs;
    
    while (1) {
        /* Get current state */
        SystemState_t state = system_get_state();
        
        /* Get sensor data */
        if (xQueuePeek(g_sensor_queue, &sensors, 0) != pdTRUE) {
            /* Use last known values from state */
            memcpy(&sensors, &state.sensors, sizeof(SensorData_t));
        }
        
        uint32_t now_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
        
        /* Only control if system is running */
        if (state.mode == SYSTEM_RUNNING) {
            /* Apply AI adjustment if valid */
            int16_t pressure_setpoint = state.setpoints.pressure_setpoint_cbar;
            if (state.ai.valid && state.ai.confidence > 50) {
                pressure_setpoint += state.ai.pressure_adjustment;
            }
            
            /* Pressure PID */
            fixed16_t pressure_sp = FLOAT_TO_FIXED16(pressure_setpoint / 100.0f);
            fixed16_t pressure_pv = FLOAT_TO_FIXED16(sensors.pressure_tank_cbar / 100.0f);
            fixed16_t motor_output = pid_compute(&pid_pressure, pressure_sp, pressure_pv);
            
            /* Convert to PWM (0-1000) */
            int32_t pwm = (FIXED16_TO_FLOAT(motor_output) + 0.5f) * PWM_OUTPUT_MAX;
            if (pwm < PWM_OUTPUT_MIN) pwm = PWM_OUTPUT_MIN;
            if (pwm > PWM_OUTPUT_MAX) pwm = PWM_OUTPUT_MAX;
            
            outputs.motor_pwm = (uint16_t)pwm;
            
            /* Update PSA cycle */
            psa_update(now_ms, state.setpoints.psa_cycle_time_ms);
            
            /* Get valve states */
            psa_get_valve_states(&outputs.valve_a_pwm, &outputs.valve_b_pwm);
            
        } else if (state.mode == SYSTEM_STANDBY || state.mode == SYSTEM_ALARM) {
            /* Safe idle state */
            outputs.motor_pwm = 0;
            outputs.valve_a_pwm = 0;
            outputs.valve_b_pwm = 0;
            
            /* Reset PIDs */
            pid_reset(&pid_pressure);
            pid_reset(&pid_flow);
            
        } else {
            /* Shutdown/fault state */
            outputs.motor_pwm = 0;
            outputs.valve_a_pwm = 0;
            outputs.valve_b_pwm = 0;
            outputs.emergency_stop = true;
        }
        
        outputs.timestamp_ms = now_ms;
        
        /* Update global state */
        system_update_outputs(&outputs);
        
        /* Send to actuators via queue */
        xQueueOverwrite(g_control_queue, &outputs);
        
        /* Wait for next period (1ms) */
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(CONTROL_TASK_PERIOD_MS));
    }
}

/**
 * Get current control output
 */
ControlOutput_t control_get_output(void)
{
    ControlOutput_t output;
    xQueuePeek(g_control_queue, &output, 0);
    return output;
}

/**
 * Set control setpoints
 */
void control_set_setpoint(Setpoint_t sp)
{
    /* This would update setpoints through system state */
}
