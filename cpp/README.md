# π-Flight Controller - C Implementation

Auto-generated C implementation of the π-Flight adaptive PID controller, trained using MCTS + Neural-Guided DSL Search.

## 📊 Performance Metrics

- **Best Score**: 3.658 (harmonic mean across 6 trajectories)
- **Verified Score**: 3.160 (on extreme test scenarios)
- **Training Iterations**: 2800
- **Test Trajectories**: zigzag3d, lemniscate3d, random_wp, spiral_in_out, stairs, coupled_surface
- **Generated**: 2025-10-18 19:57:06
- **Verified**: 2025-10-21 16:51:52

## 📁 Files

- **`pi_flight_controller.h`**: Header file with controller API
- **`pi_flight_controller.c`**: Implementation of the 5-rule segmented controller
- **`example.c`**: Demo program showing usage examples
- **`Makefile`**: Build configuration for GCC

## 🚀 Quick Start

### Build
```bash
cd cpp
make
```

### Run Example
```bash
./example
```

### Clean
```bash
make clean
```

## 🔧 Integration

### Step 1: Include the header
```c
#include "pi_flight_controller.h"
```

### Step 2: Initialize controller (optional for this version)
```c
piflight_init();
```

### Step 3: In your control loop
```c
PiFlightState state;
PIDGains gains;

// Update state from your sensors/estimators
state.err_d_pitch = ...;  // Derivative error in pitch
state.err_i_pitch = ...;  // Integral error in pitch
state.ang_vel_x = ...;    // Angular velocity around x-axis

// Compute adaptive PID gains
piflight_compute_gains(&state, &gains);

// Use gains.P, gains.I, gains.D in your PID controller
float control_output = gains.P * error + gains.I * integral + gains.D * derivative;
```

## 📐 Controller Logic

The controller implements **5 rules** evaluated sequentially:

| Rule | Condition | Action | Purpose |
|------|-----------|--------|---------|
| 1 | `1 > 0.0805` (always) | P=0.9179, I=1.9055, D=1.0921 | Baseline gains |
| 2 | `1 > 0.1` (always) | P=0.9179, I=1.9055, D=1.0921 | Baseline reinforcement |
| 3 | `err_d_pitch < 1.5` | P=1.574 | Increase P for better tracking |
| 4 | `err_i_pitch > 1.0627` | P=0.9179, I=1.9055, D=1.0921 | Anti-windup: reset when integral error is large |
| 5 | `ang_vel_x > 1.0096` | P=2.1452, I=0.815, D=0.7451 | Aggressive maneuvers: high P, lower I/D |

**Note**: Rules are evaluated in order, and later rules can override gains from earlier rules (last-write-wins semantics).

## 🎯 Key Features

✅ **Zero dependencies**: Pure C11, no external libraries  
✅ **Real-time ready**: Deterministic execution, suitable for embedded systems  
✅ **Proven performance**: Tested on 6 complex trajectories with wind disturbances  
✅ **Adaptive behavior**: Automatically adjusts PID gains based on flight state  
✅ **Easy integration**: Simple API with clear input/output structures

## 📦 Creating a Static Library

To integrate into your own project:

```bash
make libpiflight.a
```

Then link with:
```bash
gcc -o my_program my_program.c -L. -lpiflight
```

## 🔬 Technical Details

- **Input Variables**:
  - `err_d_pitch`: Derivative of pitch error (rad/s)
  - `err_i_pitch`: Integral of pitch error (rad·s)
  - `ang_vel_x`: Angular velocity around roll axis (rad/s)

- **Output Gains**:
  - `P`: Proportional gain (range: 0.815 - 2.145)
  - `I`: Integral gain (range: 0.815 - 1.906)
  - `D`: Derivative gain (range: 0.745 - 1.092)

- **Execution Time**: O(1) - constant time (5 condition checks)
- **Memory Footprint**: < 100 bytes (stateless controller)

## 📝 License

This controller implementation is part of the π-Flight project.  
See the main project repository for licensing details.

## 🤝 Citation

If you use this controller in your research, please cite the π-Flight paper:

```bibtex
@article{piflight2025,
  title={π-Flight: Neural-Guided Program Synthesis for Adaptive Drone Control},
  author={...},
  year={2025}
}
```

---

Generated from `01_pi_flight/results/best_program.json`
