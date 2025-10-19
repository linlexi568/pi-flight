"""Decision Tree model for gain scheduling using sklearn."""

import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from typing import Tuple, Optional


class DTGainScheduler:
    """Decision Tree for predicting PID gain multipliers from state features."""
    
    def __init__(self, 
                 state_dim: int = 20,
                 max_depth: int = 10,
                 min_samples_split: int = 20,
                 min_samples_leaf: int = 10,
                 p_range: Tuple[float, float] = (0.6, 2.5),
                 i_range: Tuple[float, float] = (0.5, 2.0),
                 d_range: Tuple[float, float] = (0.5, 2.0)):
        """
        Args:
            state_dim: Dimension of state features
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
            p/i/d_range: Output ranges for gain multipliers
        """
        self.state_dim = state_dim
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.p_range = p_range
        self.i_range = i_range
        self.d_range = d_range
        
        # Three separate trees for P, I, D
        self.tree_p = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        self.tree_i = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=43
        )
        self.tree_d = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=44
        )
        
        self.is_trained = False
    
    def fit(self, states: np.ndarray, gains: np.ndarray) -> dict:
        """Train the decision trees.
        
        Args:
            states: (N, state_dim) state features
            gains: (N, 3) target gain multipliers [P, I, D]
        
        Returns:
            Training info dict
        """
        assert states.shape[0] == gains.shape[0], "Mismatch in sample count"
        assert gains.shape[1] == 3, "Gains must be (N, 3) for P/I/D"
        
        # Clip gains to valid ranges
        gains_clipped = gains.copy()
        gains_clipped[:, 0] = np.clip(gains[:, 0], self.p_range[0], self.p_range[1])
        gains_clipped[:, 1] = np.clip(gains[:, 1], self.i_range[0], self.i_range[1])
        gains_clipped[:, 2] = np.clip(gains[:, 2], self.d_range[0], self.d_range[1])
        
        # Train separate trees
        self.tree_p.fit(states, gains_clipped[:, 0])
        self.tree_i.fit(states, gains_clipped[:, 1])
        self.tree_d.fit(states, gains_clipped[:, 2])
        
        self.is_trained = True
        
        # Compute training scores
        score_p = self.tree_p.score(states, gains_clipped[:, 0])
        score_i = self.tree_i.score(states, gains_clipped[:, 1])
        score_d = self.tree_d.score(states, gains_clipped[:, 2])
        
        info = {
            'n_samples': states.shape[0],
            'score_p': score_p,
            'score_i': score_i,
            'score_d': score_d,
            'score_avg': (score_p + score_i + score_d) / 3,
            'tree_depth_p': self.tree_p.get_depth(),
            'tree_depth_i': self.tree_i.get_depth(),
            'tree_depth_d': self.tree_d.get_depth(),
            'tree_leaves_p': self.tree_p.get_n_leaves(),
            'tree_leaves_i': self.tree_i.get_n_leaves(),
            'tree_leaves_d': self.tree_d.get_n_leaves(),
        }
        
        return info
    
    def predict(self, states: np.ndarray) -> np.ndarray:
        """Predict gain multipliers.
        
        Args:
            states: (N, state_dim) or (state_dim,) state features
        
        Returns:
            (N, 3) or (3,) gain multipliers [P, I, D]
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        
        single = False
        if states.ndim == 1:
            states = states.reshape(1, -1)
            single = True
        
        p_pred = self.tree_p.predict(states)
        i_pred = self.tree_i.predict(states)
        d_pred = self.tree_d.predict(states)
        
        # Clip to ranges
        p_pred = np.clip(p_pred, self.p_range[0], self.p_range[1])
        i_pred = np.clip(i_pred, self.i_range[0], self.i_range[1])
        d_pred = np.clip(d_pred, self.d_range[0], self.d_range[1])
        
        gains = np.stack([p_pred, i_pred, d_pred], axis=-1)
        
        return gains[0] if single else gains
    
    def save(self, path: str):
        """Save model to file."""
        state = {
            'tree_p': self.tree_p,
            'tree_i': self.tree_i,
            'tree_d': self.tree_d,
            'state_dim': self.state_dim,
            'max_depth': self.max_depth,
            'p_range': self.p_range,
            'i_range': self.i_range,
            'd_range': self.d_range,
            'is_trained': self.is_trained,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load model from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.tree_p = state['tree_p']
        self.tree_i = state['tree_i']
        self.tree_d = state['tree_d']
        self.state_dim = state['state_dim']
        self.max_depth = state['max_depth']
        self.p_range = state['p_range']
        self.i_range = state['i_range']
        self.d_range = state['d_range']
        self.is_trained = state['is_trained']
    
    def get_tree_stats(self) -> dict:
        """Get statistics about the trees."""
        if not self.is_trained:
            return {}
        
        return {
            'total_nodes': (self.tree_p.tree_.node_count + 
                           self.tree_i.tree_.node_count + 
                           self.tree_d.tree_.node_count),
            'total_leaves': (self.tree_p.get_n_leaves() + 
                            self.tree_i.get_n_leaves() + 
                            self.tree_d.get_n_leaves()),
            'avg_depth': (self.tree_p.get_depth() + 
                         self.tree_i.get_depth() + 
                         self.tree_d.get_depth()) / 3,
        }


if __name__ == '__main__':
    # Test
    model = DTGainScheduler(state_dim=20, max_depth=8)
    
    # Generate dummy data
    np.random.seed(42)
    states = np.random.randn(1000, 20)
    gains = np.random.uniform([0.7, 0.6, 0.6], [2.0, 1.8, 1.8], size=(1000, 3))
    
    # Train
    info = model.fit(states, gains)
    print("Training info:", info)
    
    # Predict
    test_states = np.random.randn(10, 20)
    preds = model.predict(test_states)
    print("Predictions shape:", preds.shape)
    print("Sample predictions:", preds[:3])
    
    # Save and load
    model.save('test_dt_model.pkl')
    model2 = DTGainScheduler(state_dim=20)
    model2.load('test_dt_model.pkl')
    preds2 = model2.predict(test_states)
    assert np.allclose(preds, preds2), "Save/load mismatch"
    print("Save/load test passed!")
