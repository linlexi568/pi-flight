"""Collect training data for Decision Tree by running PI-Flight controller."""

import sys
import os
import argparse
import numpy as np
import json
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from test import SimulationTester
from gym_pybullet_drones.utils.enums import DroneModel
import pybullet as p


def collect_data_from_pilight(
    program_path: str,
    traj_names: list,
    duration: float = 10,
    output_path: str = None,
    log_skip: int = 2,
    disturbance: str = None
):
    """Collect state-gain pairs by running PI-Flight controller.
    
    Args:
        program_path: Path to PI-Flight program JSON
        traj_names: List of trajectory names to collect data from
        duration: Duration of each trajectory
        output_path: Where to save the collected data (.npz)
        log_skip: Logging frequency (every N simulation steps)
        disturbance: Disturbance type (e.g., 'mild_wind')
    
    Returns:
        dict with 'states' and 'gains' arrays
    """
    # Import PI-Flight controller
    try:
        from pi_flight import PiLightSegmentedPIDController, load_program_json
    except ImportError:
        sys.path.insert(0, str(ROOT_DIR / '01_pi_flight'))
        from segmented_controller import PiLightSegmentedPIDController
        from serialization import load_program_json
    
    # Load program
    program = load_program_json(program_path)
    print(f"[Collect] Loaded program with {len(program['rules'])} rules from {program_path}")
    
    all_states = []
    all_gains = []
    
    for traj_name in traj_names:
        print(f"\n[Collect] Trajectory: {traj_name}")
        
        # Create controller
        controller = PiLightSegmentedPIDController(
            drone_model=DroneModel.CF2X,
            program=program,
            compose_by_gain=True,
            clip_D=1.2
        )
        
        # Create tester
        tester = SimulationTester(
            controller=controller,
            trajectory_name=traj_name,
            duration_sec=duration,
            gui=False,
            log_history=True,
            log_skip=log_skip,
            reward_profile='pilight_boost',
            disturbance_type=disturbance
        )
        
        # Run simulation
        reward = tester.run()
        print(f"[Collect]   Reward: {reward:.4f}")
        
        # Extract logged data
        history = tester.get_history()
        
        # For each timestep, extract state features and actual gains used
        for i in range(len(history['timestamp'])):
            # Position error
            target_pos = np.array([
                history['target_x'][i],
                history['target_y'][i],
                history['target_z'][i]
            ])
            cur_pos = np.array([
                history['x'][i],
                history['y'][i],
                history['z'][i]
            ])
            pos_e = target_pos - cur_pos
            
            # Velocity
            cur_vel = np.array([
                history['vx'][i],
                history['vy'][i],
                history['vz'][i]
            ])
            
            # Orientation
            cur_quat = np.array([
                history['qx'][i],
                history['qy'][i],
                history['qz'][i],
                history['qw'][i]
            ])
            roll, pitch, _ = p.getEulerFromQuaternion(cur_quat)
            
            # Angular velocity
            cur_ang_vel = np.array([
                history['wx'][i],
                history['wy'][i],
                history['wz'][i]
            ])
            
            # Integral terms (approximation - would need controller state)
            # For simplicity, use zeros or cumulative errors
            int_pos = np.zeros(3)  # TODO: extract from controller if available
            int_rpy = np.zeros(3)
            
            # Actual gains used (from controller)
            # PI-Flight logs the effective gains used
            if 'gain_p_torque' in history:
                gain_p = np.mean(history['gain_p_torque'][i])
                gain_i = np.mean(history['gain_i_torque'][i])
                gain_d = np.mean(history['gain_d_torque'][i])
            else:
                # Fallback: use default gains as baseline
                gain_p = 1.0
                gain_i = 1.0
                gain_d = 1.0
            
            # Build state feature (20-dim, same as GSN)
            state = np.array([
                pos_e[0], pos_e[1], pos_e[2],
                cur_vel[0], cur_vel[1], cur_vel[2],
                roll, pitch,
                cur_ang_vel[0], cur_ang_vel[1], cur_ang_vel[2],
                int_rpy[0], int_rpy[1], int_rpy[2],
                int_pos[0], int_pos[1], int_pos[2],
                gain_p, gain_i, gain_d
            ], dtype=np.float32)
            
            # Target gains (normalized multipliers relative to base gains)
            # For now, use the actual gains as they represent the effective multipliers
            gains = np.array([gain_p, gain_i, gain_d], dtype=np.float32)
            
            all_states.append(state)
            all_gains.append(gains)
        
        print(f"[Collect]   Collected {len(history['timestamp'])} samples")
    
    # Convert to arrays
    states = np.array(all_states)
    gains = np.array(all_gains)
    
    print(f"\n[Collect] Total samples: {len(states)}")
    print(f"[Collect] State shape: {states.shape}")
    print(f"[Collect] Gains shape: {gains.shape}")
    print(f"[Collect] Gain stats (P/I/D):")
    print(f"[Collect]   Mean: {gains.mean(axis=0)}")
    print(f"[Collect]   Std:  {gains.std(axis=0)}")
    print(f"[Collect]   Min:  {gains.min(axis=0)}")
    print(f"[Collect]   Max:  {gains.max(axis=0)}")
    
    # Save data
    if output_path:
        np.savez(output_path, states=states, gains=gains,
                 meta=json.dumps({
                     'program': program_path,
                     'trajectories': traj_names,
                     'duration': duration,
                     'disturbance': disturbance,
                     'state_dim': states.shape[1],
                     'n_samples': len(states)
                 }))
        print(f"[Collect] Saved to {output_path}")
    
    return {'states': states, 'gains': gains}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect training data for Decision Tree')
    parser.add_argument('--program', type=str, 
                       default='01_pi_flight/results/best_program.json',
                       help='Path to PI-Flight program JSON')
    parser.add_argument('--traj_list', type=str, nargs='+',
                       default=['zigzag3d', 'lemniscate3d', 'random_wp', 
                               'spiral_in_out', 'stairs', 'coupled_surface'],
                       help='List of trajectories to collect data from')
    parser.add_argument('--duration', type=float, default=10,
                       help='Duration of each trajectory (seconds)')
    parser.add_argument('--output', type=str,
                       default='04_decision_tree/data/dt_training_data.npz',
                       help='Output path for collected data')
    parser.add_argument('--log-skip', type=int, default=2,
                       help='Log every N simulation steps')
    parser.add_argument('--disturbance', type=str, default='mild_wind',
                       help='Disturbance type')
    
    args = parser.parse_args()
    
    data = collect_data_from_pilight(
        program_path=args.program,
        traj_names=args.traj_list,
        duration=args.duration,
        output_path=args.output,
        log_skip=args.log_skip,
        disturbance=args.disturbance
    )
    
    print("\n[Collect] Data collection complete!")
