# π-Flight C++ Controller

Auto-generated C++ implementation of the π-Flight synthesized PID controller.

## Overview

This controller was automatically synthesized using MCTS + Neural Network guided search over a domain-specific language (DSL) for segmented PID controllers. It achieved a verified performance score of **3.16** (harmonic mean) across 5 challenging trajectories.

**Training Details:**
- Iterations: 2800
- Trajectories: zigzag3d, lemniscate3d, spiral_in_out, stairs, coupled_surface
- Disturbance: mild_wind
- Verified: 2025-10-21

## Files

- `PiFlightController.h` - Controller interface
- `PiFlightController.cpp` - Controller implementation (5 rules)
- `example.cpp` - Usage demonstration
- `CMakeLists.txt` - Build configuration

## Controller Structure

The controller implements **5 rules** that adapt PID gains based on:
- **Derivative pitch error** (`err_d_pitch`)
- **Integral pitch error** (`err_i_pitch`)
- **Angular velocity X** (`ang_vel_x`)

### Rules Summary

1. **Rule 1-2**: Baseline gains (always active)
   - P=0.9179, I=1.9055, D=1.0921

2. **Rule 3**: Low derivative error → Higher P gain
   - `if err_d_pitch < 1.5 then P=1.574`

3. **Rule 4**: High integral error → Reset to baseline
   - `if err_i_pitch > 1.0627 then P=0.9179, I=1.9055, D=1.0921`

4. **Rule 5**: High angular velocity → Aggressive tuning
   - `if ang_vel_x > 1.0096 then P=2.1452, I=0.815, D=0.7451`

## Build Instructions

### Linux/macOS

```bash
mkdir build
cd build
cmake ..
make
./piflight_example
```

### Windows (Visual Studio)

```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
.\Release\piflight_example.exe
```

### Windows (MinGW)

```powershell
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
.\piflight_example.exe
```

## Usage Example

```cpp
#include "PiFlightController.h"

using namespace piflight;

int main() {
    // Create controller
    PiFlightController controller;
    
    // Prepare state
    ControllerState state;
    state.err_d_pitch = 0.5;
    state.err_i_pitch = 0.3;
    state.ang_vel_x = 0.2;
    state.dt = 0.01;
    
    // Compute gains
    PIDGains gains = controller.computeGains(state);
    
    // Apply PID control
    double control_output = gains.P * state.err_pitch 
                          + gains.I * state.err_i_pitch 
                          + gains.D * state.err_d_pitch;
    
    return 0;
}
```

## Integration Guide

### Step 1: Sensor Data Collection
Collect the following from your flight controller:
- Pitch error derivative (`err_d_pitch`)
- Pitch error integral (`err_i_pitch`)
- Angular velocity around X-axis (`ang_vel_x`)

### Step 2: State Preparation
```cpp
ControllerState state;
state.err_d_pitch = /* derivative of pitch error */;
state.err_i_pitch = /* accumulated integral of pitch error */;
state.ang_vel_x = /* gyroscope reading on X-axis */;
state.dt = /* control loop period, e.g., 0.01s */;
```

### Step 3: Gain Computation
```cpp
PIDGains gains = controller.computeGains(state);
```

### Step 4: Apply Control
```cpp
double output = gains.P * current_error 
              + gains.I * integrated_error 
              + gains.D * derivative_error;
```

## Performance

Verified on stress test trajectories (2025-10-21):
- `coupled_surface`: 3.14
- `zigzag3d`: 3.05
- `lemniscate3d`: 3.31
- `spiral_in_out`: 2.98
- `stairs`: 3.35

**Harmonic Mean: 3.16**

## Requirements

- C++14 or later
- CMake 3.10+
- No external dependencies (header-only math library)

## License

Generated from π-Flight research project.

## Citation

If you use this controller in your research, please cite:

```bibtex
@article{piflight2025,
  title={π-Flight: Neural-Guided Program Synthesis for Drone Control},
  author={[Your Name]},
  year={2025}
}
```

## Notes

- This is a **stateless controller** - no internal memory between calls
- Rules are evaluated **sequentially** - later rules can override earlier ones
- Designed for **240Hz control loops** but adaptable to other frequencies
- Optimized for **mild wind disturbances**
- Best performance on **aggressive 3D trajectories**

## Troubleshooting

**Q: Gains seem inconsistent?**
A: The controller uses rule-based switching. Check which rules are triggering for your state values.

**Q: Performance differs from Python version?**
A: Ensure your sensor calibration and error computation match the training setup.

**Q: Can I modify the rules?**
A: Yes, but the thresholds were optimized via MCTS. Manual changes may degrade performance.

## Contact

For questions about the synthesis method or controller integration, please open an issue in the π-Flight repository.
