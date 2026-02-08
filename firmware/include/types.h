/*
 * AI-Oxygen Concentrator Data Types
 * ==================================
 * 
 * Core data structures for sensor readings, control outputs, and system state.
 */

#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <stdbool.h>

/* ============ FIXED-POINT TYPES ============ */

typedef int32_t fixed16_t;  /* Q16.16 fixed-point */
typedef int16_t fixed8_t;   /* Q8.8 fixed-point */

#define FIXED16_ONE     (1 << 16)
#define FIXED8_ONE      (1 << 8)

#define FLOAT_TO_FIXED16(f)     ((fixed16_t)((f) * FIXED16_ONE))
#define FIXED16_TO_FLOAT(x)     ((float)(x) / FIXED16_ONE)
#define FLOAT_TO_FIXED8(f)      ((fixed8_t)((f) * FIXED8_ONE))
#define FIXED8_TO_FLOAT(x)      ((float)(x) / FIXED8_ONE)

/* ============ SENSOR DATA ============ */

/**
 * Raw sensor readings from ADC
 */
typedef struct {
    uint16_t pressure_tank;      /* ADC counts */
    uint16_t pressure_bed_a;
    uint16_t pressure_bed_b;
    uint16_t o2_concentration;
    uint16_t flow_rate;
    uint16_t temp_motor;
    uint16_t temp_ambient;
    uint16_t motor_current;
    uint32_t timestamp_ms;
} SensorRaw_t;

/**
 * Calibrated sensor readings
 */
typedef struct {
    int16_t pressure_tank_cbar;      /* Pressure in centibars (bar * 100) */
    int16_t pressure_bed_a_cbar;
    int16_t pressure_bed_b_cbar;
    int16_t o2_purity_permille;      /* O2 % * 10 (e.g., 930 = 93.0%) */
    int16_t flow_rate_dlpm;          /* Flow in deci-LPM (LPM * 10) */
    int16_t temp_motor_dC;           /* Temperature in deci-Celsius */
    int16_t temp_ambient_dC;
    int16_t motor_current_mA;        /* Current in milliamps */
    uint32_t timestamp_ms;
    uint8_t validity_flags;          /* Bit field for sensor validity */
} SensorData_t;

/* Validity flag bits */
#define SENSOR_VALID_PRESSURE_TANK  (1 << 0)
#define SENSOR_VALID_PRESSURE_A     (1 << 1)
#define SENSOR_VALID_PRESSURE_B     (1 << 2)
#define SENSOR_VALID_O2             (1 << 3)
#define SENSOR_VALID_FLOW           (1 << 4)
#define SENSOR_VALID_TEMP_MOTOR     (1 << 5)
#define SENSOR_VALID_TEMP_AMBIENT   (1 << 6)
#define SENSOR_VALID_CURRENT        (1 << 7)
#define SENSOR_VALID_ALL            (0xFF)

/* ============ CONTROL DATA ============ */

/**
 * Control setpoints
 */
typedef struct {
    int16_t pressure_setpoint_cbar;  /* Target pressure (centibars) */
    int16_t flow_setpoint_dlpm;      /* Target flow (deci-LPM) */
    int16_t purity_setpoint_permille; /* Target purity (permille) */
    uint16_t psa_cycle_time_ms;      /* PSA half-cycle duration */
} Setpoint_t;

/**
 * Control outputs
 */
typedef struct {
    uint16_t motor_pwm;              /* PWM duty (0-1000 = 0-100%) */
    uint16_t valve_a_pwm;            /* Valve A PWM */
    uint16_t valve_b_pwm;            /* Valve B PWM */
    bool emergency_stop;             /* Emergency stop active */
    uint32_t timestamp_ms;
} ControlOutput_t;

/**
 * PID controller state
 */
typedef struct {
    fixed16_t kp;
    fixed16_t ki;
    fixed16_t kd;
    fixed16_t integral;
    fixed16_t prev_error;
    fixed16_t output;
    bool saturated;
} PIDState_t;

/* ============ PSA STATE ============ */

typedef enum {
    PSA_PHASE_ADSORB_A = 0,
    PSA_PHASE_EQUALIZE_AB = 1,
    PSA_PHASE_ADSORB_B = 2,
    PSA_PHASE_EQUALIZE_BA = 3
} PSAPhase_t;

typedef struct {
    PSAPhase_t current_phase;
    uint32_t phase_start_time_ms;
    uint32_t cycle_count;
} PSAState_t;

/* ============ SAFETY STATE ============ */

typedef enum {
    SAFETY_NORMAL = 0,
    SAFETY_WARNING = 1,
    SAFETY_ALARM = 2,
    SAFETY_CRITICAL = 3,
    SAFETY_SHUTDOWN = 4
} SafetyLevel_t;

typedef enum {
    CONSTRAINT_NONE = 0,
    CONSTRAINT_PRESSURE_HIGH = (1 << 0),
    CONSTRAINT_PRESSURE_LOW = (1 << 1),
    CONSTRAINT_PURITY_LOW = (1 << 2),
    CONSTRAINT_TEMP_HIGH = (1 << 3),
    CONSTRAINT_FLOW_HIGH = (1 << 4),
    CONSTRAINT_ACTUATOR_RATE = (1 << 5),
    CONSTRAINT_SENSOR_FAULT = (1 << 6),
    CONSTRAINT_COMM_TIMEOUT = (1 << 7)
} SafetyConstraint_t;

typedef struct {
    SafetyLevel_t level;
    uint8_t active_constraints;      /* Bitmask of SafetyConstraint_t */
    bool emergency_stop_active;
    uint32_t alarm_count;
    uint32_t intervention_count;
    uint32_t last_check_time_ms;
} SafetyStatus_t;

/* ============ AI STATE ============ */

typedef struct {
    uint8_t recommended_action;      /* Action index [0-8] */
    int16_t pressure_adjustment;     /* Suggested pressure delta */
    int16_t flow_adjustment;         /* Suggested flow delta */
    int16_t cycle_adjustment;        /* Suggested cycle time delta */
    uint8_t confidence;              /* Confidence 0-100 */
    bool valid;                      /* Recommendation is valid */
    uint32_t inference_time_us;      /* Last inference duration */
} AIRecommendation_t;

typedef struct {
    uint8_t health_score;            /* 0-100 health percentage */
    uint8_t anomaly_score;           /* 0-100 anomaly likelihood */
    bool health_warning;
    bool anomaly_detected;
    uint32_t last_update_time_ms;
} HealthStatus_t;

/* ============ SYSTEM STATE ============ */

typedef enum {
    SYSTEM_INITIALIZING = 0,
    SYSTEM_SELF_TEST = 1,
    SYSTEM_STANDBY = 2,
    SYSTEM_RUNNING = 3,
    SYSTEM_ALARM = 4,
    SYSTEM_SHUTDOWN = 5,
    SYSTEM_FAULT = 6
} SystemMode_t;

typedef struct {
    SystemMode_t mode;
    SensorData_t sensors;
    Setpoint_t setpoints;
    ControlOutput_t outputs;
    PSAState_t psa;
    SafetyStatus_t safety;
    AIRecommendation_t ai;
    HealthStatus_t health;
    uint32_t uptime_seconds;
    uint32_t energy_wh;
} SystemState_t;

/* ============ DATA LOGGING ============ */

typedef struct {
    uint32_t timestamp_ms;
    int16_t pressure_cbar;
    int16_t purity_permille;
    int16_t flow_dlpm;
    int16_t temp_motor_dC;
    uint16_t motor_pwm;
    uint8_t psa_phase;
    uint8_t safety_level;
    uint8_t health_score;
} LogEntry_t;

/* ============ COMMUNICATION ============ */

typedef struct {
    uint8_t command_id;
    uint8_t payload[16];
    uint8_t payload_length;
} CommandPacket_t;

typedef struct {
    uint8_t message_type;
    uint8_t payload[32];
    uint8_t payload_length;
    uint32_t timestamp_ms;
} TelemetryPacket_t;

/* Message types */
#define MSG_TYPE_STATUS     0x01
#define MSG_TYPE_SENSORS    0x02
#define MSG_TYPE_ALARM      0x03
#define MSG_TYPE_CONFIG     0x04
#define MSG_TYPE_LOG        0x05

/* Command IDs */
#define CMD_START           0x01
#define CMD_STOP            0x02
#define CMD_SET_FLOW        0x03
#define CMD_SET_PRESSURE    0x04
#define CMD_EMERGENCY_STOP  0x05
#define CMD_RESET           0x06
#define CMD_GET_STATUS      0x07

#endif /* TYPES_H */
