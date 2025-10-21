/**
 * @file example.c
 * @brief Example usage of the π-Flight controller
 * 
 * Demonstrates how to integrate the adaptive PID controller
 * into a flight control loop.
 */

#include "pi_flight_controller.h"
#include <stdio.h>

int main(void) {
    PiFlightState state;
    PIDGains gains;

    // Initialize controller
    piflight_init();

    printf("=== π-Flight Adaptive PID Controller Demo ===\n");
    printf("Version: %s\n\n", piflight_get_version());

    // Example 1: Normal flight conditions
    printf("Test Case 1: Normal flight\n");
    state.err_d_pitch = 0.5f;
    state.err_i_pitch = 0.3f;
    state.ang_vel_x = 0.2f;
    piflight_compute_gains(&state, &gains);
    printf("  State: err_d_pitch=%.2f, err_i_pitch=%.2f, ang_vel_x=%.2f\n",
           state.err_d_pitch, state.err_i_pitch, state.ang_vel_x);
    printf("  Gains: P=%.4f, I=%.4f, D=%.4f\n\n", gains.P, gains.I, gains.D);

    // Example 2: High roll rate (aggressive maneuver)
    printf("Test Case 2: High roll rate (aggressive maneuver)\n");
    state.err_d_pitch = 0.8f;
    state.err_i_pitch = 0.5f;
    state.ang_vel_x = 1.5f;  // Triggers Rule 5
    piflight_compute_gains(&state, &gains);
    printf("  State: err_d_pitch=%.2f, err_i_pitch=%.2f, ang_vel_x=%.2f\n",
           state.err_d_pitch, state.err_i_pitch, state.ang_vel_x);
    printf("  Gains: P=%.4f, I=%.4f, D=%.4f\n", gains.P, gains.I, gains.D);
    printf("  -> Rule 5 active: High P for quick response\n\n");

    // Example 3: Large integral error (anti-windup)
    printf("Test Case 3: Large integral error\n");
    state.err_d_pitch = 0.4f;
    state.err_i_pitch = 2.0f;  // Triggers Rule 4
    state.ang_vel_x = 0.3f;
    piflight_compute_gains(&state, &gains);
    printf("  State: err_d_pitch=%.2f, err_i_pitch=%.2f, ang_vel_x=%.2f\n",
           state.err_d_pitch, state.err_i_pitch, state.ang_vel_x);
    printf("  Gains: P=%.4f, I=%.4f, D=%.4f\n", gains.P, gains.I, gains.D);
    printf("  -> Rule 4 active: Reset to baseline (anti-windup)\n\n");

    // Example 4: Low derivative error
    printf("Test Case 4: Low derivative error\n");
    state.err_d_pitch = 1.2f;  // Triggers Rule 3
    state.err_i_pitch = 0.6f;
    state.ang_vel_x = 0.4f;
    piflight_compute_gains(&state, &gains);
    printf("  State: err_d_pitch=%.2f, err_i_pitch=%.2f, ang_vel_x=%.2f\n",
           state.err_d_pitch, state.err_i_pitch, state.ang_vel_x);
    printf("  Gains: P=%.4f, I=%.4f, D=%.4f\n", gains.P, gains.I, gains.D);
    printf("  -> Rule 3 active: Increased P for better tracking\n\n");

    printf("=== Integration Example ===\n");
    printf("In your control loop:\n");
    printf("  1. Compute state.err_d_pitch, state.err_i_pitch, state.ang_vel_x\n");
    printf("  2. Call piflight_compute_gains(&state, &gains)\n");
    printf("  3. Use gains.P, gains.I, gains.D in your PID controller\n");
    printf("  4. Repeat each control step (e.g., 240 Hz)\n\n");

    return 0;
}
