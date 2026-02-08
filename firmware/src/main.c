/*
 * AI-Oxygen Concentrator Main Entry Point
 * ========================================
 * 
 * FreeRTOS application with real-time control tasks.
 * 
 * Task Architecture:
 * - Safety Task: Highest priority, 1kHz watchdog
 * - Sensor Task: 1kHz ADC acquisition
 * - Control Task: 1kHz PID control loop
 * - AI Task: 10Hz TFLite inference
 * - Logging Task: 1Hz data logging
 * - Comm Task: Async communication handling
 */

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

#include "config.h"
#include "types.h"

/* External task functions */
extern void sensor_task(void *pvParameters);
extern void control_task(void *pvParameters);
extern void safety_task(void *pvParameters);
extern void ai_task(void *pvParameters);
extern void logging_task(void *pvParameters);
extern void comm_task(void *pvParameters);

/* Global system state (protected by mutex) */
static SystemState_t g_system_state;
static SemaphoreHandle_t g_state_mutex;

/* Inter-task queues */
QueueHandle_t g_sensor_queue;
QueueHandle_t g_control_queue;
QueueHandle_t g_log_queue;

/* Task handles */
static TaskHandle_t h_safety_task;
static TaskHandle_t h_sensor_task;
static TaskHandle_t h_control_task;
static TaskHandle_t h_ai_task;
static TaskHandle_t h_logging_task;
static TaskHandle_t h_comm_task;


/**
 * Initialize system state to safe defaults
 */
static void init_system_state(void)
{
    memset(&g_system_state, 0, sizeof(SystemState_t));
    
    g_system_state.mode = SYSTEM_INITIALIZING;
    
    /* Default setpoints */
    g_system_state.setpoints.pressure_setpoint_cbar = 200;  /* 2.0 bar */
    g_system_state.setpoints.flow_setpoint_dlpm = 30;       /* 3.0 LPM */
    g_system_state.setpoints.purity_setpoint_permille = 930; /* 93.0% */
    g_system_state.setpoints.psa_cycle_time_ms = PSA_CYCLE_TIME_DEFAULT_MS;
    
    /* Safety starts in normal state */
    g_system_state.safety.level = SAFETY_NORMAL;
    g_system_state.safety.active_constraints = CONSTRAINT_NONE;
    g_system_state.safety.emergency_stop_active = false;
    
    /* PSA starts in phase A */
    g_system_state.psa.current_phase = PSA_PHASE_ADSORB_A;
    g_system_state.psa.cycle_count = 0;
    
    /* AI not valid until first inference */
    g_system_state.ai.valid = false;
    
    printf("[MAIN] System state initialized\n");
}


/**
 * Initialize hardware peripherals
 */
static bool init_hardware(void)
{
    printf("[MAIN] Initializing hardware...\n");
    
    /* TODO: Initialize ADC */
    /* TODO: Initialize PWM */
    /* TODO: Initialize GPIO */
    /* TODO: Initialize UART */
    /* TODO: Initialize I2C */
    
    printf("[MAIN] Hardware initialization complete\n");
    return true;
}


/**
 * Perform system self-test
 */
static bool run_self_test(void)
{
    printf("[MAIN] Running self-test...\n");
    
    g_system_state.mode = SYSTEM_SELF_TEST;
    
    /* TODO: Test sensor readings are in valid range */
    /* TODO: Test actuator response */
    /* TODO: Test safety shutdown */
    /* TODO: Verify AI model loaded */
    
    printf("[MAIN] Self-test passed\n");
    return true;
}


/**
 * Create FreeRTOS queues
 */
static void create_queues(void)
{
    g_sensor_queue = xQueueCreate(QUEUE_SENSOR_SIZE, sizeof(SensorData_t));
    g_control_queue = xQueueCreate(QUEUE_CONTROL_SIZE, sizeof(ControlOutput_t));
    g_log_queue = xQueueCreate(QUEUE_LOG_SIZE, sizeof(LogEntry_t));
    
    if (g_sensor_queue == NULL || g_control_queue == NULL || g_log_queue == NULL) {
        printf("[MAIN] ERROR: Failed to create queues\n");
        while(1);  /* Halt on queue creation failure */
    }
    
    printf("[MAIN] Queues created\n");
}


/**
 * Create FreeRTOS tasks
 */
static void create_tasks(void)
{
    BaseType_t ret;
    
    /* Safety task - highest priority */
    ret = xTaskCreate(
        safety_task,
        "safety",
        STACK_SIZE_SAFETY,
        NULL,
        PRIORITY_SAFETY,
        &h_safety_task
    );
    configASSERT(ret == pdPASS);
    
    /* Sensor task */
    ret = xTaskCreate(
        sensor_task,
        "sensor",
        STACK_SIZE_SENSOR,
        NULL,
        PRIORITY_SENSOR,
        &h_sensor_task
    );
    configASSERT(ret == pdPASS);
    
    /* Control task */
    ret = xTaskCreate(
        control_task,
        "control",
        STACK_SIZE_CONTROL,
        NULL,
        PRIORITY_CONTROL,
        &h_control_task
    );
    configASSERT(ret == pdPASS);
    
    /* AI task - needs more stack for TFLite */
    ret = xTaskCreate(
        ai_task,
        "ai",
        STACK_SIZE_AI,
        NULL,
        PRIORITY_AI,
        &h_ai_task
    );
    configASSERT(ret == pdPASS);
    
    /* Logging task */
    ret = xTaskCreate(
        logging_task,
        "logging",
        STACK_SIZE_LOGGING,
        NULL,
        PRIORITY_LOGGING,
        &h_logging_task
    );
    configASSERT(ret == pdPASS);
    
    /* Communication task */
    ret = xTaskCreate(
        comm_task,
        "comm",
        STACK_SIZE_COMM,
        NULL,
        PRIORITY_COMM,
        &h_comm_task
    );
    configASSERT(ret == pdPASS);
    
    printf("[MAIN] All tasks created\n");
}


/**
 * Get current system state (thread-safe)
 */
SystemState_t system_get_state(void)
{
    SystemState_t state;
    
    if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        memcpy(&state, &g_system_state, sizeof(SystemState_t));
        xSemaphoreGive(g_state_mutex);
    } else {
        /* Return last known state if mutex not available */
        memcpy(&state, &g_system_state, sizeof(SystemState_t));
    }
    
    return state;
}


/**
 * Update system state (thread-safe)
 */
void system_update_sensors(const SensorData_t *sensors)
{
    if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        memcpy(&g_system_state.sensors, sensors, sizeof(SensorData_t));
        xSemaphoreGive(g_state_mutex);
    }
}


void system_update_outputs(const ControlOutput_t *outputs)
{
    if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        memcpy(&g_system_state.outputs, outputs, sizeof(ControlOutput_t));
        xSemaphoreGive(g_state_mutex);
    }
}


void system_update_safety(const SafetyStatus_t *safety)
{
    if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        memcpy(&g_system_state.safety, safety, sizeof(SafetyStatus_t));
        
        /* Update system mode based on safety */
        if (safety->level == SAFETY_SHUTDOWN) {
            g_system_state.mode = SYSTEM_SHUTDOWN;
        } else if (safety->level >= SAFETY_ALARM) {
            g_system_state.mode = SYSTEM_ALARM;
        }
        
        xSemaphoreGive(g_state_mutex);
    }
}


void system_update_ai(const AIRecommendation_t *ai)
{
    if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        memcpy(&g_system_state.ai, ai, sizeof(AIRecommendation_t));
        xSemaphoreGive(g_state_mutex);
    }
}


void system_set_mode(SystemMode_t mode)
{
    if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        g_system_state.mode = mode;
        xSemaphoreGive(g_state_mutex);
    }
}


/**
 * Application entry point
 */
void app_main(void)
{
    printf("\n");
    printf("========================================\n");
    printf("  AI-Oxygen Concentrator v%d.%d.%d\n",
           FIRMWARE_VERSION_MAJOR,
           FIRMWARE_VERSION_MINOR,
           FIRMWARE_VERSION_PATCH);
    printf("========================================\n\n");
    
    /* Create state mutex */
    g_state_mutex = xSemaphoreCreateMutex();
    configASSERT(g_state_mutex != NULL);
    
    /* Initialize system */
    init_system_state();
    
    if (!init_hardware()) {
        printf("[MAIN] ERROR: Hardware initialization failed\n");
        g_system_state.mode = SYSTEM_FAULT;
        while(1);
    }
    
    /* Create queues and tasks */
    create_queues();
    create_tasks();
    
    /* Run self-test */
    if (!run_self_test()) {
        printf("[MAIN] ERROR: Self-test failed\n");
        g_system_state.mode = SYSTEM_FAULT;
        while(1);
    }
    
    /* Enter standby mode */
    g_system_state.mode = SYSTEM_STANDBY;
    printf("[MAIN] System ready - entering standby\n");
    
    /* Main loop - just monitor system health */
    while (1) {
        /* Update uptime */
        g_system_state.uptime_seconds = xTaskGetTickCount() / configTICK_RATE_HZ;
        
        /* Print status every 10 seconds */
        static uint32_t last_status = 0;
        if (g_system_state.uptime_seconds - last_status >= 10) {
            last_status = g_system_state.uptime_seconds;
            
            printf("[STATUS] Mode=%d, P=%.2f bar, O2=%.1f%%, Safety=%d\n",
                   g_system_state.mode,
                   g_system_state.sensors.pressure_tank_cbar / 100.0f,
                   g_system_state.sensors.o2_purity_permille / 10.0f,
                   g_system_state.safety.level);
        }
        
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
