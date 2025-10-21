/**
 * @file PiFlightController.h
 * @brief π-Flight Synthesized PID Controller
 * 
 * Auto-generated from best_program.json
 * Generated: 2025-10-21
 * Performance: 3.16 (harmonic mean over 5 trajectories)
 * Training iterations: 2800
 */

#ifndef PIFLIGHT_CONTROLLER_H
#define PIFLIGHT_CONTROLLER_H

#include <cmath>
#include <array>

namespace piflight {

/**
 * @brief State structure containing sensor inputs
 */
struct ControllerState {
    // Position errors
    double err_x;
    double err_y;
    double err_z;
    
    // Velocity errors
    double err_vx;
    double err_vy;
    double err_vz;
    
    // Attitude errors (roll, pitch, yaw)
    double err_roll;
    double err_pitch;
    double err_yaw;
    
    // Integral errors
    double err_i_roll;
    double err_i_pitch;
    double err_i_yaw;
    
    // Derivative errors
    double err_d_roll;
    double err_d_pitch;
    double err_d_yaw;
    
    // Angular velocities
    double ang_vel_x;
    double ang_vel_y;
    double ang_vel_z;
    
    // Time delta (for integration/differentiation)
    double dt;
};

/**
 * @brief PID gains structure
 */
struct PIDGains {
    double P;
    double I;
    double D;
    
    PIDGains() : P(1.0), I(0.0), D(0.0) {}
    PIDGains(double p, double i, double d) : P(p), I(i), D(d) {}
};

/**
 * @brief π-Flight Controller class
 * 
 * Implements a rule-based PID gain scheduler synthesized by MCTS+NN.
 * The controller adapts PID gains based on current flight state.
 */
class PiFlightController {
public:
    PiFlightController();
    ~PiFlightController() = default;
    
    /**
     * @brief Compute PID gains for current state
     * @param state Current controller state (errors, velocities, etc.)
     * @return PID gains (P, I, D)
     */
    PIDGains computeGains(const ControllerState& state) const;
    
    /**
     * @brief Reset controller internal state
     */
    void reset();
    
    /**
     * @brief Get controller metadata
     */
    const char* getMetadata() const;
    
private:
    // Default gains (fallback if no rule matches)
    PIDGains default_gains_;
};

} // namespace piflight

#endif // PIFLIGHT_CONTROLLER_H
