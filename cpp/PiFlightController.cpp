/**
 * @file PiFlightController.cpp
 * @brief π-Flight Synthesized PID Controller Implementation
 * 
 * Auto-generated from best_program.json
 * 
 * Program Structure:
 * - 5 rules with conditions on errors and angular velocities
 * - Rule-based gain scheduling for adaptive PID control
 * 
 * Training Performance:
 * - Best score: 3.16 (harmonic mean)
 * - Trajectories: zigzag3d, lemniscate3d, spiral_in_out, stairs, coupled_surface
 * - Verified: 2025-10-21 16:51:52
 */

#include "PiFlightController.h"

namespace piflight {

PiFlightController::PiFlightController() 
    : default_gains_(1.0, 0.0, 0.0) {
}

void PiFlightController::reset() {
    // No internal state to reset in this stateless controller
}

const char* PiFlightController::getMetadata() const {
    return "π-Flight Controller v1.0 | Score: 3.16 | Rules: 5 | Trained: 2025-10-18";
}

PIDGains PiFlightController::computeGains(const ControllerState& state) const {
    PIDGains gains = default_gains_;
    
    // Rule 1: if (1 > 0.0805) then P=0.9179, I=1.9055, D=1.0921
    // Note: This is always true (constant condition)
    if (1.0 > 0.0805) {
        gains.P = 0.9179;
        gains.I = 1.9055;
        gains.D = 1.0921;
    }
    
    // Rule 2: if (1 > 0.1) then P=0.9179, I=1.9055, D=1.0921
    // Note: This is always true (constant condition)
    if (1.0 > 0.1) {
        gains.P = 0.9179;
        gains.I = 1.9055;
        gains.D = 1.0921;
    }
    
    // Rule 3: if (err_d_pitch < 1.5) then P=1.574
    if (state.err_d_pitch < 1.5) {
        gains.P = 1.574;
    }
    
    // Rule 4: if (err_i_pitch > 1.0627) then P=0.9179, I=1.9055, D=1.0921
    if (state.err_i_pitch > 1.0627) {
        gains.P = 0.9179;
        gains.I = 1.9055;
        gains.D = 1.0921;
    }
    
    // Rule 5: if (ang_vel_x > 1.0096) then P=2.1452, I=0.815, D=0.7451
    if (state.ang_vel_x > 1.0096) {
        gains.P = 2.1452;
        gains.I = 0.815;
        gains.D = 0.7451;
    }
    
    return gains;
}

} // namespace piflight
