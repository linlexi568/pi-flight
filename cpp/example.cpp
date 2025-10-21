/**
 * @file example.cpp
 * @brief Example usage of PiFlightController
 * 
 * Demonstrates how to integrate the synthesized controller
 * into a flight control loop.
 */

#include "PiFlightController.h"
#include <iostream>
#include <iomanip>

using namespace piflight;

int main() {
    std::cout << "=== π-Flight Controller Example ===" << std::endl;
    std::cout << std::endl;
    
    // Create controller instance
    PiFlightController controller;
    
    // Print metadata
    std::cout << "Controller Info: " << controller.getMetadata() << std::endl;
    std::cout << std::endl;
    
    // Example 1: Low angular velocity scenario
    std::cout << "--- Scenario 1: Low angular velocity ---" << std::endl;
    ControllerState state1;
    state1.err_d_pitch = 0.5;
    state1.err_i_pitch = 0.3;
    state1.ang_vel_x = 0.2;
    state1.dt = 0.01;
    
    PIDGains gains1 = controller.computeGains(state1);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Inputs:" << std::endl;
    std::cout << "  err_d_pitch = " << state1.err_d_pitch << std::endl;
    std::cout << "  err_i_pitch = " << state1.err_i_pitch << std::endl;
    std::cout << "  ang_vel_x   = " << state1.ang_vel_x << std::endl;
    std::cout << "Gains:" << std::endl;
    std::cout << "  P = " << gains1.P << std::endl;
    std::cout << "  I = " << gains1.I << std::endl;
    std::cout << "  D = " << gains1.D << std::endl;
    std::cout << std::endl;
    
    // Example 2: High angular velocity scenario
    std::cout << "--- Scenario 2: High angular velocity ---" << std::endl;
    ControllerState state2;
    state2.err_d_pitch = 2.0;
    state2.err_i_pitch = 0.5;
    state2.ang_vel_x = 1.5;  // > 1.0096 threshold
    state2.dt = 0.01;
    
    PIDGains gains2 = controller.computeGains(state2);
    std::cout << "Inputs:" << std::endl;
    std::cout << "  err_d_pitch = " << state2.err_d_pitch << std::endl;
    std::cout << "  err_i_pitch = " << state2.err_i_pitch << std::endl;
    std::cout << "  ang_vel_x   = " << state2.ang_vel_x << std::endl;
    std::cout << "Gains:" << std::endl;
    std::cout << "  P = " << gains2.P << std::endl;
    std::cout << "  I = " << gains2.I << std::endl;
    std::cout << "  D = " << gains2.D << std::endl;
    std::cout << std::endl;
    
    // Example 3: High integral error scenario
    std::cout << "--- Scenario 3: High integral error ---" << std::endl;
    ControllerState state3;
    state3.err_d_pitch = 0.8;
    state3.err_i_pitch = 1.5;  // > 1.0627 threshold
    state3.ang_vel_x = 0.5;
    state3.dt = 0.01;
    
    PIDGains gains3 = controller.computeGains(state3);
    std::cout << "Inputs:" << std::endl;
    std::cout << "  err_d_pitch = " << state3.err_d_pitch << std::endl;
    std::cout << "  err_i_pitch = " << state3.err_i_pitch << std::endl;
    std::cout << "  ang_vel_x   = " << state3.ang_vel_x << std::endl;
    std::cout << "Gains:" << std::endl;
    std::cout << "  P = " << gains3.P << std::endl;
    std::cout << "  I = " << gains3.I << std::endl;
    std::cout << "  D = " << gains3.D << std::endl;
    std::cout << std::endl;
    
    // Example integration into control loop
    std::cout << "--- Pseudo Control Loop ---" << std::endl;
    std::cout << "for each timestep:" << std::endl;
    std::cout << "  1. Read sensor data (IMU, position, etc.)" << std::endl;
    std::cout << "  2. Compute errors (position, attitude, derivatives)" << std::endl;
    std::cout << "  3. state.err_d_pitch = current_derivative_error" << std::endl;
    std::cout << "  4. state.err_i_pitch = accumulated_integral_error" << std::endl;
    std::cout << "  5. state.ang_vel_x = gyroscope_reading_x" << std::endl;
    std::cout << "  6. gains = controller.computeGains(state)" << std::endl;
    std::cout << "  7. control_output = gains.P * err + gains.I * err_i + gains.D * err_d" << std::endl;
    std::cout << "  8. Send control_output to motors" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Example Complete ===" << std::endl;
    
    return 0;
}
