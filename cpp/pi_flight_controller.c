/**
 * @file pi_flight_controller.c
 * @brief π-Flight Adaptive PID Controller - Implementation
 * 
 * Auto-generated from best_program.json (2025-10-18 19:57:06)
 * 
 * Controller Logic:
 *   5 rules evaluated sequentially, each checking a condition and
 *   updating PID gains if the condition is met. Later rules can
 *   override earlier ones.
 * 
 * Rule Summary:
 *   Rule 1: if (1 > 0.0805) -> P=0.9179, I=1.9055, D=1.0921
 *   Rule 2: if (1 > 0.1)    -> P=0.9179, I=1.9055, D=1.0921
 *   Rule 3: if (err_d_pitch < 1.5) -> P=1.574
 *   Rule 4: if (err_i_pitch > 1.0627) -> P=0.9179, I=1.9055, D=1.0921
 *   Rule 5: if (ang_vel_x > 1.0096) -> P=2.1452, I=0.815, D=0.7451
 */

#include "pi_flight_controller.h"

// Controller version and metadata
static const char* VERSION_STRING = 
    "π-Flight v1.0 | Score: 3.658 | Iters: 2800 | Date: 2025-10-18";

void piflight_init(void) {
    // Reserved for future initialization (e.g., filter state, logging)
}

void piflight_compute_gains(const PiFlightState* state, PIDGains* gains) {
    // Initialize gains to default values (will be overridden by rules)
    gains->P = 0.0f;
    gains->I = 0.0f;
    gains->D = 0.0f;

    // Rule 1: if (1 > 0.0805) then P=0.9179, I=1.9055, D=1.0921
    // Condition is always true (constant comparison), acts as baseline
    if (1.0f > 0.0805f) {
        gains->P = 0.9179f;
        gains->I = 1.9055f;
        gains->D = 1.0921f;
    }

    // Rule 2: if (1 > 0.1) then P=0.9179, I=1.9055, D=1.0921
    // Also always true, redundant baseline reinforcement
    if (1.0f > 0.1f) {
        gains->P = 0.9179f;
        gains->I = 1.9055f;
        gains->D = 1.0921f;
    }

    // Rule 3: if (err_d_pitch < 1.5) then P=1.574
    // Increase P gain when derivative pitch error is moderate
    if (state->err_d_pitch < 1.5f) {
        gains->P = 1.574f;
        // I and D remain from previous rules
    }

    // Rule 4: if (err_i_pitch > 1.0627) then P=0.9179, I=1.9055, D=1.0921
    // Reset to baseline when integral error is large (anti-windup behavior)
    if (state->err_i_pitch > 1.0627f) {
        gains->P = 0.9179f;
        gains->I = 1.9055f;
        gains->D = 1.0921f;
    }

    // Rule 5: if (ang_vel_x > 1.0096) then P=2.1452, I=0.815, D=0.7451
    // High P, lower I/D when roll rate is high (aggressive maneuvering)
    if (state->ang_vel_x > 1.0096f) {
        gains->P = 2.1452f;
        gains->I = 0.815f;
        gains->D = 0.7451f;
    }
}

const char* piflight_get_version(void) {
    return VERSION_STRING;
}
