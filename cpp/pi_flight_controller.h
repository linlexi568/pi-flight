/**
 * @file pi_flight_controller.h
 * @brief π-Flight Adaptive PID Controller - C Implementation
 * 
 * Auto-generated from best_program.json
 * Trained with MCTS + Neural-Guided DSL Search
 * 
 * Performance Metrics:
 *   - Best Score: 3.658 (harmonic mean across 6 trajectories)
 *   - Verified Score: 3.160 (on extreme test scenarios)
 *   - Training Iterations: 2800
 *   - Test Trajectories: zigzag3d, lemniscate3d, random_wp, 
 *                        spiral_in_out, stairs, coupled_surface
 * 
 * Generated: 2025-10-18 19:57:06
 * Verified: 2025-10-21 16:51:52
 */

#ifndef PI_FLIGHT_CONTROLLER_H
#define PI_FLIGHT_CONTROLLER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Controller input state structure
 * 
 * Contains all state variables needed by the rule-based controller
 */
typedef struct {
    float err_d_pitch;      // Derivative error in pitch
    float err_i_pitch;      // Integral error in pitch
    float ang_vel_x;        // Angular velocity around x-axis (roll rate)
} PiFlightState;

/**
 * @brief PID gains output structure
 */
typedef struct {
    float P;  // Proportional gain
    float I;  // Integral gain
    float D;  // Derivative gain
} PIDGains;

/**
 * @brief Initialize the controller (if needed for future extensions)
 */
void piflight_init(void);

/**
 * @brief Compute adaptive PID gains based on current state
 * 
 * Implements a 5-rule segmented controller that adapts PID gains
 * according to flight state conditions.
 * 
 * @param state Current flight state (errors and angular velocities)
 * @param gains Output PID gains structure (will be populated)
 * 
 * @note Rules are evaluated in order, and later rules can override
 *       gains set by earlier rules (last-write-wins semantics)
 */
void piflight_compute_gains(const PiFlightState* state, PIDGains* gains);

/**
 * @brief Get controller metadata (for diagnostics)
 * 
 * @return String describing the controller version and performance
 */
const char* piflight_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // PI_FLIGHT_CONTROLLER_H
