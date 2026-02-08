/*
 * Sensor Acquisition Task
 * =======================
 * 
 * High-frequency sensor reading at 1kHz.
 * - ADC sampling with DMA
 * - Calibration and filtering
 * - Validity checking
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "config.h"
#include "types.h"

/* External references */
extern QueueHandle_t g_sensor_queue;
extern void system_update_sensors(const SensorData_t *sensors);

/* Low-pass filter state */
static int32_t filter_state[8] = {0};
#define FILTER_ALPHA  16  /* 1/16 = 0.0625 smoothing */

/* Temperature lookup table (NTC thermistor) */
static const int16_t temp_lookup[TEMP_LOOKUP_SIZE] = {
    /* ADC value -> Temperature (dC) */
    1200, 1100, 1000, 900, 800, 700, 600, 550, 500, 450,
    400, 350, 300, 250, 200, 150, 100, 50, 0, -50, -100
};

/**
 * Read raw ADC values (platform-specific)
 */
static void read_adc_all(SensorRaw_t *raw)
{
    /* TODO: Implement actual ADC reading for target platform */
    /* For now, placeholder with simulated values */
    
    raw->pressure_tank = 2048;     /* Mid-scale */
    raw->pressure_bed_a = 2048;
    raw->pressure_bed_b = 1024;
    raw->o2_concentration = 380;   /* ~93% */
    raw->flow_rate = 1228;         /* ~3 LPM */
    raw->temp_motor = 1800;        /* ~45°C */
    raw->temp_ambient = 2000;      /* ~25°C */
    raw->motor_current = 2500;     /* ~5A */
    raw->timestamp_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
}

/**
 * Apply low-pass filter
 */
static int16_t apply_filter(int32_t *state, int16_t new_value)
{
    *state = *state + (((int32_t)new_value - *state) / FILTER_ALPHA);
    return (int16_t)*state;
}

/**
 * Convert ADC to pressure (centibars)
 */
static int16_t adc_to_pressure(uint16_t adc)
{
    if (adc <= PRESSURE_ADC_MIN) return 0;
    if (adc >= PRESSURE_ADC_MAX) return PRESSURE_RANGE_BAR;
    
    int32_t scaled = (int32_t)(adc - PRESSURE_ADC_MIN) * PRESSURE_RANGE_BAR;
    return (int16_t)(scaled / (PRESSURE_ADC_MAX - PRESSURE_ADC_MIN));
}

/**
 * Convert ADC to O2 percentage (permille)
 */
static int16_t adc_to_o2(uint16_t adc)
{
    if (adc >= O2_ADC_MAX) return O2_RANGE_PERCENT;
    
    int32_t scaled = (int32_t)adc * O2_RANGE_PERCENT;
    return (int16_t)(scaled / O2_ADC_MAX);
}

/**
 * Convert ADC to flow rate (deci-LPM)
 */
static int16_t adc_to_flow(uint16_t adc)
{
    int32_t scaled = (int32_t)adc * FLOW_RANGE_LPM;
    return (int16_t)(scaled / FLOW_ADC_MAX);
}

/**
 * Convert ADC to temperature using lookup table
 */
static int16_t adc_to_temp(uint16_t adc)
{
    /* Linear interpolation between lookup points */
    int idx = adc / (4096 / TEMP_LOOKUP_SIZE);
    if (idx >= TEMP_LOOKUP_SIZE - 1) idx = TEMP_LOOKUP_SIZE - 2;
    if (idx < 0) idx = 0;
    
    return temp_lookup[idx];
}

/**
 * Convert ADC to motor current (mA)
 */
static int16_t adc_to_current(uint16_t adc)
{
    /* ACS712: Vout = 2.5V + (I * 0.185V/A) */
    int32_t voltage_mv = (adc * 3300) / 4095;
    int32_t current_ma = ((voltage_mv - CURRENT_ZERO_OFFSET_MV) * 1000) / CURRENT_SENSITIVITY_MV_A;
    return (int16_t)current_ma;
}

/**
 * Check sensor validity
 */
static uint8_t check_validity(const SensorData_t *data)
{
    uint8_t flags = 0;
    
    /* Pressure tank: 0-5 bar expected */
    if (data->pressure_tank_cbar >= 0 && data->pressure_tank_cbar <= 500)
        flags |= SENSOR_VALID_PRESSURE_TANK;
    
    /* Pressure beds: similar */
    if (data->pressure_bed_a_cbar >= 0 && data->pressure_bed_a_cbar <= 500)
        flags |= SENSOR_VALID_PRESSURE_A;
    
    if (data->pressure_bed_b_cbar >= 0 && data->pressure_bed_b_cbar <= 500)
        flags |= SENSOR_VALID_PRESSURE_B;
    
    /* O2: 0-100% */
    if (data->o2_purity_permille >= 0 && data->o2_purity_permille <= 1000)
        flags |= SENSOR_VALID_O2;
    
    /* Flow: 0-15 LPM */
    if (data->flow_rate_dlpm >= 0 && data->flow_rate_dlpm <= 150)
        flags |= SENSOR_VALID_FLOW;
    
    /* Temperature: -20 to 120°C */
    if (data->temp_motor_dC >= -200 && data->temp_motor_dC <= 1200)
        flags |= SENSOR_VALID_TEMP_MOTOR;
    
    if (data->temp_ambient_dC >= -200 && data->temp_ambient_dC <= 600)
        flags |= SENSOR_VALID_TEMP_AMBIENT;
    
    /* Current: 0-25A */
    if (data->motor_current_mA >= 0 && data->motor_current_mA <= 25000)
        flags |= SENSOR_VALID_CURRENT;
    
    return flags;
}

/**
 * Sensor acquisition task
 */
void sensor_task(void *pvParameters)
{
    (void)pvParameters;
    
    TickType_t last_wake = xTaskGetTickCount();
    SensorRaw_t raw;
    SensorData_t data;
    
    while (1) {
        /* Read all ADCs */
        read_adc_all(&raw);
        
        /* Convert and calibrate */
        data.pressure_tank_cbar = adc_to_pressure(raw.pressure_tank);
        data.pressure_bed_a_cbar = adc_to_pressure(raw.pressure_bed_a);
        data.pressure_bed_b_cbar = adc_to_pressure(raw.pressure_bed_b);
        data.o2_purity_permille = adc_to_o2(raw.o2_concentration);
        data.flow_rate_dlpm = adc_to_flow(raw.flow_rate);
        data.temp_motor_dC = adc_to_temp(raw.temp_motor);
        data.temp_ambient_dC = adc_to_temp(raw.temp_ambient);
        data.motor_current_mA = adc_to_current(raw.motor_current);
        
        /* Apply filters */
        data.pressure_tank_cbar = apply_filter(&filter_state[0], data.pressure_tank_cbar);
        data.o2_purity_permille = apply_filter(&filter_state[3], data.o2_purity_permille);
        data.flow_rate_dlpm = apply_filter(&filter_state[4], data.flow_rate_dlpm);
        
        /* Set timestamp */
        data.timestamp_ms = raw.timestamp_ms;
        
        /* Check validity */
        data.validity_flags = check_validity(&data);
        
        /* Update global state */
        system_update_sensors(&data);
        
        /* Send to queue (non-blocking) */
        xQueueOverwrite(g_sensor_queue, &data);
        
        /* Wait for next period (1ms) */
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SENSOR_TASK_PERIOD_MS));
    }
}

/**
 * Get latest sensor data (for other modules)
 */
SensorData_t sensor_get_latest(void)
{
    SensorData_t data;
    xQueuePeek(g_sensor_queue, &data, 0);
    return data;
}

/**
 * Check if specific sensor is valid
 */
bool sensor_is_valid(uint8_t sensor_flag)
{
    SensorData_t data;
    xQueuePeek(g_sensor_queue, &data, 0);
    return (data.validity_flags & sensor_flag) != 0;
}
