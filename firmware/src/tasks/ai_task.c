/*
 * AI Task
 * =======
 * 
 * TensorFlow Lite Micro inference at 10Hz.
 * - Runs DQN for action recommendation
 * - Runs LSTM for health prediction
 * - All outputs validated by safety layer
 */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "config.h"
#include "types.h"

/* TensorFlow Lite Micro includes */
/* #include "tensorflow/lite/micro/all_ops_resolver.h" */
/* #include "tensorflow/lite/micro/micro_interpreter.h" */
/* #include "tensorflow/lite/schema/schema_generated.h" */

/* External references */
extern SystemState_t system_get_state(void);
extern void system_update_ai(const AIRecommendation_t *ai);

/* TFLite arena */
static uint8_t tensor_arena[TFLITE_ARENA_SIZE];

/* Model data (embedded from .tflite file) */
/* extern const unsigned char dqn_model_tflite[]; */
/* extern const unsigned int dqn_model_tflite_len; */

/* State buffer for inference */
static float state_buffer[AI_STATE_DIM];

/* LSTM sliding window */
static float lstm_window[60][8];  /* 60 timesteps, 8 features */
static int lstm_window_idx = 0;

/**
 * Prepare state vector from sensor data
 */
static void prepare_state_vector(const SystemState_t *state)
{
    /* Normalize inputs to [0, 1] or [-1, 1] range */
    
    /* 0: Tank pressure (0.5-3.0 bar) */
    state_buffer[0] = (state->sensors.pressure_tank_cbar / 100.0f - 0.5f) / 2.5f;
    
    /* 1: Bed A pressure */
    state_buffer[1] = (state->sensors.pressure_bed_a_cbar / 100.0f - 0.5f) / 2.5f;
    
    /* 2: Bed B pressure */
    state_buffer[2] = (state->sensors.pressure_bed_b_cbar / 100.0f - 0.5f) / 2.5f;
    
    /* 3: O2 purity (21-99%) */
    state_buffer[3] = (state->sensors.o2_purity_permille / 10.0f - 21.0f) / 78.0f;
    
    /* 4: Purity rate (assumed 0 for now) */
    state_buffer[4] = 0.0f;
    
    /* 5: Flow rate (0-15 LPM) */
    state_buffer[5] = state->sensors.flow_rate_dlpm / 150.0f;
    
    /* 6: SpO2 (assumed from external, using purity as proxy) */
    state_buffer[6] = (state->sensors.o2_purity_permille / 10.0f - 50.0f) / 50.0f;
    
    /* 7: SpO2 trend (assumed 0) */
    state_buffer[7] = 0.0f;
    
    /* 8: Motor temperature (0-100°C) */
    state_buffer[8] = state->sensors.temp_motor_dC / 1000.0f;
    
    /* 9: Compressor health */
    state_buffer[9] = state->health.health_score / 100.0f;
    
    /* 10: Zeolite efficiency */
    state_buffer[10] = 0.95f;  /* Assumed */
    
    /* 11: Power (0-500W) */
    float power = state->sensors.motor_current_mA * 24.0f / 1000.0f;
    state_buffer[11] = power / 500.0f;
    
    /* 12: PSA phase (0-3) */
    state_buffer[12] = (float)state->psa.current_phase / 3.0f;
    
    /* 13: Cycle time (10-60s) */
    state_buffer[13] = (state->setpoints.psa_cycle_time_ms / 1000.0f - 10.0f) / 50.0f;
}

/**
 * Run DQN inference
 */
static uint8_t run_dqn_inference(float *q_values)
{
    /* TODO: Actual TFLite Micro inference */
    /* For now, return default action (maintain) */
    
    /*
    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        return 2;  // Default: maintain
    }
    
    float* output = interpreter->output(0)->data.f;
    memcpy(q_values, output, AI_ACTION_DIM * sizeof(float));
    
    // Find argmax
    int best_action = 0;
    float best_value = q_values[0];
    for (int i = 1; i < AI_ACTION_DIM; i++) {
        if (q_values[i] > best_value) {
            best_value = q_values[i];
            best_action = i;
        }
    }
    return best_action;
    */
    
    /* Placeholder: simple rule-based fallback */
    float pressure = state_buffer[0] * 2.5f + 0.5f;  /* De-normalize */
    
    if (pressure > 2.8f) return 0;      /* Decrease pressure */
    if (pressure < 1.5f) return 4;      /* Increase pressure */
    return 2;                            /* Maintain */
}

/**
 * Convert action to control adjustments
 */
static void action_to_adjustments(uint8_t action, AIRecommendation_t *ai)
{
    ai->recommended_action = action;
    
    switch (action) {
        case 0:  /* Decrease pressure large */
            ai->pressure_adjustment = -20;  /* -0.2 bar in centibars */
            ai->flow_adjustment = 0;
            ai->cycle_adjustment = 0;
            break;
        case 1:  /* Decrease pressure small */
            ai->pressure_adjustment = -5;
            ai->flow_adjustment = 0;
            ai->cycle_adjustment = 0;
            break;
        case 2:  /* Maintain */
            ai->pressure_adjustment = 0;
            ai->flow_adjustment = 0;
            ai->cycle_adjustment = 0;
            break;
        case 3:  /* Increase pressure small */
            ai->pressure_adjustment = 5;
            ai->flow_adjustment = 0;
            ai->cycle_adjustment = 0;
            break;
        case 4:  /* Increase pressure large */
            ai->pressure_adjustment = 20;
            ai->flow_adjustment = 0;
            ai->cycle_adjustment = 0;
            break;
        case 5:  /* Decrease cycle time */
            ai->pressure_adjustment = 0;
            ai->flow_adjustment = 0;
            ai->cycle_adjustment = -2000;  /* -2 seconds in ms */
            break;
        case 6:  /* Increase cycle time */
            ai->pressure_adjustment = 0;
            ai->flow_adjustment = 0;
            ai->cycle_adjustment = 2000;
            break;
        case 7:  /* Decrease flow */
            ai->pressure_adjustment = 0;
            ai->flow_adjustment = -5;  /* -0.5 LPM in deci-LPM */
            ai->cycle_adjustment = 0;
            break;
        case 8:  /* Increase flow */
            ai->pressure_adjustment = 0;
            ai->flow_adjustment = 5;
            ai->cycle_adjustment = 0;
            break;
        default:
            ai->pressure_adjustment = 0;
            ai->flow_adjustment = 0;
            ai->cycle_adjustment = 0;
            break;
    }
}

/**
 * Update LSTM window and run health prediction
 */
static void update_health_prediction(const SystemState_t *state, HealthStatus_t *health)
{
    /* Add current features to sliding window */
    lstm_window[lstm_window_idx][0] = state->sensors.motor_current_mA / 1000.0f;
    lstm_window[lstm_window_idx][1] = state->sensors.temp_motor_dC / 100.0f;
    lstm_window[lstm_window_idx][2] = state->sensors.pressure_tank_cbar / 100.0f;
    lstm_window[lstm_window_idx][3] = state->sensors.flow_rate_dlpm / 10.0f;
    lstm_window[lstm_window_idx][4] = state->sensors.o2_purity_permille / 1000.0f;
    lstm_window[lstm_window_idx][5] = state->psa.cycle_count / 10000.0f;
    lstm_window[lstm_window_idx][6] = state->sensors.motor_current_mA * 24.0f / 1000.0f / 100.0f;
    lstm_window[lstm_window_idx][7] = 0.01f;  /* Pressure variance placeholder */
    
    lstm_window_idx = (lstm_window_idx + 1) % 60;
    
    /* TODO: Run LSTM inference when window is full */
    /* For now, use simple heuristics */
    
    /* Health degrades with temperature */
    float temp_factor = 1.0f - (state->sensors.temp_motor_dC - 400) / 500.0f;
    if (temp_factor > 1.0f) temp_factor = 1.0f;
    if (temp_factor < 0.0f) temp_factor = 0.0f;
    
    health->health_score = (uint8_t)(temp_factor * 100.0f);
    health->anomaly_score = (state->sensors.temp_motor_dC > 700) ? 80 : 10;
    health->health_warning = (health->health_score < 80);
    health->anomaly_detected = (health->anomaly_score > 70);
    health->last_update_time_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
}

/**
 * AI task main loop
 */
void ai_task(void *pvParameters)
{
    (void)pvParameters;
    
    AIRecommendation_t ai_rec;
    HealthStatus_t health;
    float q_values[AI_ACTION_DIM];
    
    TickType_t last_wake = xTaskGetTickCount();
    
    /* TODO: Initialize TFLite interpreter */
    /*
    tflite::MicroInterpreter* interpreter = ...;
    */
    
    while (1) {
        SystemState_t state = system_get_state();
        uint32_t start_time = xTaskGetTickCount();
        
        /* Only run AI if system is active */
        if (state.mode == SYSTEM_RUNNING || state.mode == SYSTEM_STANDBY) {
            
            /* Prepare input */
            prepare_state_vector(&state);
            
            /* Run DQN inference */
            uint8_t action = run_dqn_inference(q_values);
            
            /* Convert to adjustments */
            action_to_adjustments(action, &ai_rec);
            
            /* Calculate confidence from Q-value spread */
            float max_q = q_values[0];
            float sum_q = q_values[0];
            for (int i = 1; i < AI_ACTION_DIM; i++) {
                if (q_values[i] > max_q) max_q = q_values[i];
                sum_q += q_values[i];
            }
            float avg_q = sum_q / AI_ACTION_DIM;
            ai_rec.confidence = (uint8_t)((max_q - avg_q) * 100);
            if (ai_rec.confidence > 100) ai_rec.confidence = 100;
            
            ai_rec.valid = true;
            ai_rec.inference_time_us = (xTaskGetTickCount() - start_time) * 1000;
            
            /* Update health prediction */
            update_health_prediction(&state, &health);
            
            /* Send to system state */
            system_update_ai(&ai_rec);
        }
        
        /* Wait for next period (100ms) */
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(AI_TASK_PERIOD_MS));
    }
}

/**
 * Get latest AI recommendation
 */
AIRecommendation_t ai_get_recommendation(void)
{
    SystemState_t state = system_get_state();
    return state.ai;
}

/**
 * Get health score
 */
float ai_get_health_score(void)
{
    SystemState_t state = system_get_state();
    return state.health.health_score / 100.0f;
}
