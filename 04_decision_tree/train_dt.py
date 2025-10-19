"""Train Decision Tree model for PID gain scheduling."""

import argparse
import numpy as np
from pathlib import Path
import json

from dt_model import DTGainScheduler


def train_decision_tree(
    data_path: str,
    output_path: str = None,
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    test_split: float = 0.2
):
    """Train decision tree model on collected data.
    
    Args:
        data_path: Path to .npz data file
        output_path: Where to save trained model (.pkl)
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split node
        min_samples_leaf: Minimum samples at leaf
        test_split: Fraction of data for validation
    
    Returns:
        Trained model and metrics dict
    """
    print(f"\n{'='*60}")
    print("Decision Tree Training")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"[Train] Loading data from {data_path}")
    data = np.load(data_path)
    states = data['states']
    gains = data['gains']
    
    if 'meta' in data:
        meta = json.loads(str(data['meta']))
        print(f"[Train] Meta info: {meta}")
    
    print(f"[Train] Loaded {len(states)} samples")
    print(f"[Train]   State dim: {states.shape[1]}")
    print(f"[Train]   Gain stats:")
    print(f"[Train]     P: {gains[:, 0].mean():.3f} ± {gains[:, 0].std():.3f}")
    print(f"[Train]     I: {gains[:, 1].mean():.3f} ± {gains[:, 1].std():.3f}")
    print(f"[Train]     D: {gains[:, 2].mean():.3f} ± {gains[:, 2].std():.3f}")
    
    # Split train/validation
    n_samples = len(states)
    n_train = int(n_samples * (1 - test_split))
    indices = np.random.permutation(n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    states_train = states[train_idx]
    gains_train = gains[train_idx]
    states_val = states[val_idx]
    gains_val = gains[val_idx]
    
    print(f"\n[Train] Split:")
    print(f"[Train]   Training:   {len(states_train)} samples")
    print(f"[Train]   Validation: {len(states_val)} samples")
    
    # Create model
    print(f"\n[Train] Model config:")
    print(f"[Train]   Max depth: {max_depth}")
    print(f"[Train]   Min samples split: {min_samples_split}")
    print(f"[Train]   Min samples leaf: {min_samples_leaf}")
    
    model = DTGainScheduler(
        state_dim=states.shape[1],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    
    # Train
    print(f"\n[Train] Training...")
    train_info = model.fit(states_train, gains_train)
    
    print(f"\n[Train] Training results:")
    print(f"[Train]   R² scores:")
    print(f"[Train]     P: {train_info['score_p']:.4f}")
    print(f"[Train]     I: {train_info['score_i']:.4f}")
    print(f"[Train]     D: {train_info['score_d']:.4f}")
    print(f"[Train]     Avg: {train_info['score_avg']:.4f}")
    print(f"[Train]   Tree depths: P={train_info['tree_depth_p']}, "
          f"I={train_info['tree_depth_i']}, D={train_info['tree_depth_d']}")
    print(f"[Train]   Tree leaves: P={train_info['tree_leaves_p']}, "
          f"I={train_info['tree_leaves_i']}, D={train_info['tree_leaves_d']}")
    
    # Validation
    print(f"\n[Train] Validation...")
    val_pred = model.predict(states_val)
    
    # Compute validation R² scores
    from sklearn.metrics import r2_score, mean_absolute_error
    
    r2_p = r2_score(gains_val[:, 0], val_pred[:, 0])
    r2_i = r2_score(gains_val[:, 1], val_pred[:, 1])
    r2_d = r2_score(gains_val[:, 2], val_pred[:, 2])
    r2_avg = (r2_p + r2_i + r2_d) / 3
    
    mae_p = mean_absolute_error(gains_val[:, 0], val_pred[:, 0])
    mae_i = mean_absolute_error(gains_val[:, 1], val_pred[:, 1])
    mae_d = mean_absolute_error(gains_val[:, 2], val_pred[:, 2])
    
    print(f"[Train] Validation results:")
    print(f"[Train]   R² scores:")
    print(f"[Train]     P: {r2_p:.4f}")
    print(f"[Train]     I: {r2_i:.4f}")
    print(f"[Train]     D: {r2_d:.4f}")
    print(f"[Train]     Avg: {r2_avg:.4f}")
    print(f"[Train]   MAE:")
    print(f"[Train]     P: {mae_p:.4f}")
    print(f"[Train]     I: {mae_i:.4f}")
    print(f"[Train]     D: {mae_d:.4f}")
    
    # Save model
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
        print(f"\n[Train] Model saved to {output_path}")
        
        # Also save training log
        log_path = str(Path(output_path).with_suffix('.json'))
        log_data = {
            'train_info': train_info,
            'val_r2_p': float(r2_p),
            'val_r2_i': float(r2_i),
            'val_r2_d': float(r2_d),
            'val_r2_avg': float(r2_avg),
            'val_mae_p': float(mae_p),
            'val_mae_i': float(mae_i),
            'val_mae_d': float(mae_d),
            'config': {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'test_split': test_split,
                'n_train': len(states_train),
                'n_val': len(states_val),
            }
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"[Train] Training log saved to {log_path}")
    
    metrics = {
        'train_r2': train_info['score_avg'],
        'val_r2': r2_avg,
        'val_mae_avg': (mae_p + mae_i + mae_d) / 3,
        'tree_stats': model.get_tree_stats()
    }
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")
    
    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Decision Tree for gain scheduling')
    parser.add_argument('--data', type=str,
                       default='04_decision_tree/data/dt_training_data.npz',
                       help='Path to training data (.npz)')
    parser.add_argument('--output', type=str,
                       default='04_decision_tree/results/dt_model.pkl',
                       help='Output path for trained model')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum tree depth')
    parser.add_argument('--min-samples-split', type=int, default=20,
                       help='Minimum samples to split node')
    parser.add_argument('--min-samples-leaf', type=int, default=10,
                       help='Minimum samples at leaf')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Fraction of data for validation')
    
    args = parser.parse_args()
    
    model, metrics = train_decision_tree(
        data_path=args.data,
        output_path=args.output,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        test_split=args.test_split
    )
    
    print(f"\n[Summary] Final metrics:")
    print(f"  Training R²: {metrics['train_r2']:.4f}")
    print(f"  Validation R²: {metrics['val_r2']:.4f}")
    print(f"  Validation MAE: {metrics['val_mae_avg']:.4f}")
    print(f"  Total nodes: {metrics['tree_stats']['total_nodes']}")
    print(f"  Total leaves: {metrics['tree_stats']['total_leaves']}")
