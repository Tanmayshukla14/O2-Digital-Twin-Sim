/*
 * AI-Oxygen Concentrator Firmware Configuration
 * ==============================================
 * 
 * Hardware configuration and system constants.
 * Target: ESP32 / STM32 with FreeRTOS
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

/* ============ SYSTEM CONFIGURATION ============ */

#define FIRMWARE_VERSION_MAJOR  1
#define FIRMWARE_VERSION_MINOR  0
#define FIRMWARE_VERSION_PATCH  0

/* Target Platform */
#define PLATFORM_ESP32    1
#define PLATFORM_STM32    2
#define TARGET_PLATFORM   PLATFORM_ESP32

/* ============ TIMING CONFIGURATION ============ */

/* Task periods (in milliseconds) */
#define SENSOR_TASK_PERIOD_MS       1       /* 1 kHz */
#define CONTROL_TASK_PERIOD_MS      1       /* 1 kHz */
#define SAFETY_TASK_PERIOD_MS       1       /* 1 kHz */
#define AI_TASK_PERIOD_MS           100     /* 10 Hz */
#define LOGGING_TASK_PERIOD_MS      1000    /* 1 Hz */
#define COMM_TASK_PERIOD_MS         100     /* 10 Hz */

/* Watchdog timeout (ms) */
#define WATCHDOG_TIMEOUT_MS         500

/* ============ SAFETY LIMITS ============ */

/* Pressure limits (bar * 100 for integer math) */
#define PRESSURE_MIN_ALARM          50      /* 0.5 bar */
#define PRESSURE_MIN_WARN           70      /* 0.7 bar */
#define PRESSURE_MAX_WARN           280     /* 2.8 bar */
#define PRESSURE_MAX_ALARM          300     /* 3.0 bar */
#define PRESSURE_EMERGENCY_MAX      350     /* 3.5 bar (hardware relief) */

/* Purity limits (% * 10) */
#define PURITY_MIN_ALARM            870     /* 87.0% */
#define PURITY_MIN_WARN             890     /* 89.0% */

/* Temperature limits (Celsius * 10) */
#define TEMP_MAX_WARN               750     /* 75.0°C */
#define TEMP_MAX_ALARM              850     /* 85.0°C */
#define TEMP_MAX_SHUTDOWN           950     /* 95.0°C */

/* Flow limits (LPM * 10) */
#define FLOW_MAX_ALARM              100     /* 10.0 LPM */
#define FLOW_MAX_WARN               90      /* 9.0 LPM */

/* Actuator rate limit (% per 100ms) */
#define PWM_RATE_LIMIT_PERCENT      10

/* ============ HARDWARE PINS (ESP32) ============ */

#if TARGET_PLATFORM == PLATFORM_ESP32

/* ADC Pins */
#define PIN_ADC_PRESSURE_TANK       GPIO_NUM_34
#define PIN_ADC_PRESSURE_BED_A      GPIO_NUM_35
#define PIN_ADC_PRESSURE_BED_B      GPIO_NUM_32
#define PIN_ADC_O2_SENSOR           GPIO_NUM_33
#define PIN_ADC_FLOW                GPIO_NUM_36
#define PIN_ADC_TEMP_MOTOR          GPIO_NUM_39
#define PIN_ADC_TEMP_AMBIENT        GPIO_NUM_25
#define PIN_ADC_MOTOR_CURRENT       GPIO_NUM_26

/* PWM Pins */
#define PIN_PWM_MOTOR               GPIO_NUM_21
#define PIN_PWM_VALVE_A             GPIO_NUM_22
#define PIN_PWM_VALVE_B             GPIO_NUM_23

/* Digital I/O */
#define PIN_RELAY_EMERGENCY         GPIO_NUM_19
#define PIN_LED_STATUS              GPIO_NUM_2
#define PIN_LED_ALARM               GPIO_NUM_4
#define PIN_BUZZER                  GPIO_NUM_5

/* Communication */
#define PIN_UART_TX                 GPIO_NUM_17
#define PIN_UART_RX                 GPIO_NUM_16
#define PIN_I2C_SDA                 GPIO_NUM_21
#define PIN_I2C_SCL                 GPIO_NUM_22

#endif /* PLATFORM_ESP32 */

/* ============ HARDWARE PINS (STM32) ============ */

#if TARGET_PLATFORM == PLATFORM_STM32

/* STM32 pin definitions would go here */
#define PIN_ADC_PRESSURE_TANK       PA0
#define PIN_ADC_PRESSURE_BED_A      PA1
/* ... etc */

#endif /* PLATFORM_STM32 */

/* ============ ADC CALIBRATION ============ */

/* Pressure sensor: 0.5-4.5V for 0-5 bar */
#define PRESSURE_ADC_MIN            409     /* 0.5V at 12-bit */
#define PRESSURE_ADC_MAX            3686    /* 4.5V at 12-bit */
#define PRESSURE_RANGE_BAR          500     /* 0-5.0 bar (in centibars) */

/* O2 sensor: 0-1V for 0-100% */
#define O2_ADC_MIN                  0
#define O2_ADC_MAX                  819     /* 1.0V at 12-bit */
#define O2_RANGE_PERCENT            1000    /* 0-100.0% */

/* Temperature: NTC thermistor lookup */
#define TEMP_LOOKUP_SIZE            21

/* Flow sensor: 0-5V for 0-15 LPM */
#define FLOW_ADC_MIN                0
#define FLOW_ADC_MAX                4095
#define FLOW_RANGE_LPM              150     /* 0-15.0 LPM */

/* Current sensor: ACS712 5A, 185mV/A */
#define CURRENT_SENSITIVITY_MV_A    185
#define CURRENT_ZERO_OFFSET_MV      2500

/* ============ CONTROL PARAMETERS ============ */

/* PID gains (Q16.16 fixed-point) */
#define PID_KP_DEFAULT              (2 << 16)       /* 2.0 */
#define PID_KI_DEFAULT              (1 << 15)       /* 0.5 */
#define PID_KD_DEFAULT              (1 << 14)       /* 0.25 */

/* Integral limits */
#define PID_INTEGRAL_MAX            (100 << 16)
#define PID_INTEGRAL_MIN            (-100 << 16)

/* Output limits (0-100% as 0-1000) */
#define PWM_OUTPUT_MIN              0
#define PWM_OUTPUT_MAX              1000

/* ============ PSA CONFIGURATION ============ */

/* Cycle timing (ms) */
#define PSA_CYCLE_TIME_DEFAULT_MS   20000   /* 20 seconds */
#define PSA_CYCLE_TIME_MIN_MS       10000   /* 10 seconds */
#define PSA_CYCLE_TIME_MAX_MS       60000   /* 60 seconds */
#define PSA_EQUALIZATION_TIME_MS    1000    /* 1 second */

/* ============ AI CONFIGURATION ============ */

/* TensorFlow Lite Micro arena size */
#define TFLITE_ARENA_SIZE           (30 * 1024)     /* 30 KB */

/* AI inference input size */
#define AI_STATE_DIM                14
#define AI_ACTION_DIM               9

/* AI update rate */
#define AI_INFERENCE_PERIOD_MS      100     /* 10 Hz */

/* ============ COMMUNICATION ============ */

#define UART_BAUD_RATE              115200
#define BLE_ENABLED                 1
#define WIFI_ENABLED                0

/* Data logging */
#define LOG_BUFFER_SIZE             4096
#define LOG_FLUSH_INTERVAL_MS       5000

/* ============ FREERTOS CONFIGURATION ============ */

/* Task priorities (higher = more important) */
#define PRIORITY_SAFETY             (configMAX_PRIORITIES - 1)  /* 7 */
#define PRIORITY_SENSOR             (configMAX_PRIORITIES - 2)  /* 6 */
#define PRIORITY_CONTROL            (configMAX_PRIORITIES - 3)  /* 5 */
#define PRIORITY_AI                 (configMAX_PRIORITIES - 5)  /* 3 */
#define PRIORITY_LOGGING            (configMAX_PRIORITIES - 6)  /* 2 */
#define PRIORITY_COMM               (configMAX_PRIORITIES - 7)  /* 1 */

/* Task stack sizes (words) */
#define STACK_SIZE_SAFETY           1024
#define STACK_SIZE_SENSOR           1024
#define STACK_SIZE_CONTROL          2048
#define STACK_SIZE_AI               4096    /* TFLite needs more */
#define STACK_SIZE_LOGGING          2048
#define STACK_SIZE_COMM             2048

/* Queue sizes */
#define QUEUE_SENSOR_SIZE           16
#define QUEUE_CONTROL_SIZE          8
#define QUEUE_LOG_SIZE              64

#endif /* CONFIG_H */
