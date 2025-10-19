# Attention Baseline (AttnGainNet)

This folder now contains a tiny Transformer-based gain scheduler that adjusts torque PID gains at runtime. It mirrors the GSN baseline and integrates with the same evaluation/logging.

## 1) Collect dataset (state -> target gains)
- Reuse the existing collector which runs a rules program and logs the gains actually applied.
- Output default: `04_nn_baselines/data/gsn_dataset.npz`

Example (headless, short):
- python 04_nn_baselines/dataset_builder.py --duration-sec 6 --episodes 1 --output 04_nn_baselines/data/gsn_dataset.npz

Notes:
- Ensure a valid program JSON exists at `01_pi_flight/results/best_program.json` (preferred) or let the collector use the built-in example.

## 2) Train Attention model
- Trains a small Transformer encoder on sliding windows over the dataset.
- Saves best checkpoint to `04_nn_baselines/results/checkpoints/attn_best.pt`.

Example:
- python 04_nn_baselines/train_attn.py --epochs 10 --seq_len 8 --bs 128 --lr 3e-4

## 3) Evaluate (headless, multi-trajectory)
- The main headless evaluator supports the ATTN controller.

Examples:
- python main_no_gui.py --mode attn_only --traj_preset train_core --aggregate harmonic --duration_eval 8
- python main_no_gui.py --mode compare_all --traj_preset full_eval --aggregate harmonic --duration_eval 8 --gsn_ckpt 04_nn_baselines/results/checkpoints/gsn_best.pt --attn_ckpt 04_nn_baselines/results/checkpoints/attn_best.pt

Artifacts:
- Results saved under `04_nn_baselines/results/eval_attn/` per trajectory.
- Summary JSON saved to `results/summaries/` as with other baselines.

## Implementation notes
- `attn_model.py`: AttnGainNet with 2-layer Transformer encoder; outputs are mapped via sigmoid into safe P/I/D ranges.
- `attn_controller.py`: wraps `DSLPIDControl`; maintains a short feature buffer (default T=8), applies EMA smoothing, and scales torque PID baselines each step.
- `train_attn.py`: builds short sequences from the flat dataset, MSE loss to the target multipliers.

## Troubleshooting
- If imports fail, ensure `04_nn_baselines` is on `sys.path` (main_no_gui handles this). 
- If dataset is missing, run the collector first.
- If evaluation is slow, reduce `--duration_eval`, use `--traj_preset train_core`, or set fewer trajectories.
