/*
 * Safety Task
 * ===========
 * 
 * Highest priority deterministic safety watchdog.
 * - Checks all safety invariants at 1kHz
 * - Cannot be bypassed by AI
 * - Triggers emergency shutdown if needed
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "config.h"
#include "types.h"

/* External references */
extern SystemState_t system_get_state(void);
extern void system_update_safety(const SafetyStatus_t *safety);
extern void system_set_mode(SystemMode_t mode);

/* Local state */
static SafetyStatus_t safety_status;
static uint32_t alarm_start_time = 0;
static uint16_t last_pwm = 0;
static uint32_t last_pwm_time = 0;

/**
 * Check pressure constraints
 */
static uint8_t check_pressure(const SensorData_t *sensors)
{
    uint8_t constraints = CONSTRAINT_NONE;
    
    /* Tank pressure high */
    if (sensors->pressure_tank_cbar >= PRESSURE_MAX_ALARM) {
        constraints |= CONSTRAINT_PRESSURE_HIGH;
    }
    
    /* Tank pressure low */
    if (sensors->pressure_tank_cbar <= PRESSURE_MIN_ALARM) {
        constraints |= CONSTRAINT_PRESSURE_LOW;
    }
    
    return constraints;
}

/**
 * Check purity constraints
 */
static uint8_t check_purity(const SensorData_t *sensors)
{
    if (sensors->o2_purity_permille < PURITY_MIN_ALARM) {
        return CONSTRAINT_PURITY_LOW;
    }
    return CONSTRAINT_NONE;
}

/**
 * Check temperature constraints
 */
static uint8_t check_temperature(const SensorData_t *sensors)
{
    if (sensors->temp_motor_dC >= TEMP_MAX_ALARM) {
        return CONSTRAINT_TEMP_HIGH;
    }
    return CONSTRAINT_NONE;
}

/**
 * Check sensor validity
 */
static uint8_t check_sensors(const SensorData_t *sensors)
{
    /* Critical sensors must be valid */
    uint8_t critical_mask = SENSOR_VALID_PRESSURE_TANK | 
                            SENSOR_VALID_O2 | 
                            SENSOR_VALID_TEMP_MOTOR;
    
    if ((sensors->validity_flags & critical_mask) != critical_mask) {
        return CONSTRAINT_SENSOR_FAULT;
    }
    return CONSTRAINT_NONE;
}

/**
 * Check actuator rate limits
 */
static uint8_t check_actuator_rate(uint16_t current_pwm, uint32_t current_time)
{
    if (last_pwm_time == 0) {
        last_pwm = current_pwm;
        last_pwm_time = current_time;
        return CONSTRAINT_NONE;
    }
    
    uint32_t dt_ms = current_time - last_pwm_time;
    if (dt_ms < 100) {
        /* Check rate over 100ms window */
        int16_t delta = (int16_t)current_pwm - (int16_t)last_pwm;
        int16_t max_delta = (PWM_RATE_LIMIT_PERCENT * PWM_OUTPUT_MAX) / 100;
        
        if (delta > max_delta || delta < -max_delta) {
            return CONSTRAINT_ACTUATOR_RATE;
        }
    } else {
        last_pwm = current_pwm;
        last_pwm_time = current_time;
    }
    
    return CONSTRAINT_NONE;
}

/**
 * Determine safety level from constraints
 */
static SafetyLevel_t determine_level(uint8_t constraints);

/**
 * Trigger emergency stop
 */
static void trigger_emergency_stop(void)
{
    safety_status.emergency_stop_active = true;
    safety_status.level = SAFETY_SHUTDOWN;
    system_set_mode(SYSTEM_SHUTDOWN);
    
    /* TODO: Directly control GPIO for emergency relay */
    /* gpio_set_level(PIN_RELAY_EMERGENCY, 0); */
}

/**
 * Safety task main loop
 */
void safety_task(void *pvParameters)
{
    (void)pvParameters;
    
    /* Initialize */
    memset(&safety_status, 0, sizeof(SafetyStatus_t));
    safety_status.level = SAFETY_NORMAL;
    
    TickType_t last_wake = xTaskGetTickCount();
    
    while (1) {
        SystemState_t state = system_get_state();
        uint32_t now_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
        
        /* Reset constraints */
        uint8_t constraints = CONSTRAINT_NONE;
        
        /* Check all safety invariants */
        constraints |= check_pressure(&state.sensors);
        constraints |= check_purity(&state.sensors);
        constraints |= check_temperature(&state.sensors);
        constraints |= check_sensors(&state.sensors);
        constraints |= check_actuator_rate(state.outputs.motor_pwm, now_ms);
        
        /* Critical temperature check - immediate shutdown */
        if (state.sensors.temp_motor_dC >= TEMP_MAX_SHUTDOWN) {
            constraints |= CONSTRAINT_TEMP_HIGH;
            trigger_emergency_stop();
        }
        
        /* Emergency pressure check */
        if (state.sensors.pressure_tank_cbar >= PRESSURE_EMERGENCY_MAX * 100 / 100) {
            constraints |= CONSTRAINT_PRESSURE_HIGH;
            trigger_emergency_stop();
        }
        
        /* Update status */
        safety_status.active_constraints = constraints;
        safety_status.last_check_time_ms = now_ms;
        
        /* Determine level */
        if (safety_status.emergency_stop_active) {
            safety_status.level = SAFETY_SHUTDOWN;
        } else if (constraints & (CONSTRAINT_TEMP_HIGH | CONSTRAINT_SENSOR_FAULT)) {
            safety_status.level = SAFETY_CRITICAL;
        } else if (constraints != CONSTRAINT_NONE) {
            safety_status.level = SAFETY_ALARM;
            
            /* Track alarm duration */
            if (alarm_start_time == 0) {
                alarm_start_time = now_ms;
                safety_status.alarm_count++;
            }
        } else {
            safety_status.level = SAFETY_NORMAL;
            alarm_start_time = 0;
        }
        
        /* Update global state */
        system_update_safety(&safety_status);
        
        /* Wait for next period (1ms) */
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SAFETY_TASK_PERIOD_MS));
    }
}

/**
 * Get current safety status
 */
SafetyStatus_t safety_get_status(void)
{
    return safety_status;
}

/**
 * Trigger emergency stop from external call
 */
void safety_emergency_stop(void)
{
    trigger_emergency_stop();
}

/**
 * Attempt to reset emergency stop
 */
bool safety_reset(void)
{
    SystemState_t state = system_get_state();
    
    /* Only reset if temperature is safe */
    if (state.sensors.temp_motor_dC < TEMP_MAX_WARN) {
        safety_status.emergency_stop_active = false;
        safety_status.level = SAFETY_NORMAL;
        system_set_mode(SYSTEM_STANDBY);
        return true;
    }
    
    return false;
}
