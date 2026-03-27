"""
Optuna hyperparameter tuning for TabNet triage model.

Optimizes TabNet architecture and training parameters using Optuna
with macro-F1 score as the objective metric.
"""

import numpy as np
import json
import os
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "tabnet"))

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("[ERROR] Optuna not installed. Install with:")
    print("  pip install optuna")
    sys.exit(1)

from train_tabnet import load_tabnet_data
from utils import (
    scale_tabnet_features,
    compute_tabnet_class_weights,
)

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception as e:
    print("[ERROR] Failed to import TabNetClassifier from pytorch_tabnet.tab_model")
    print(f"  Root cause: {type(e).__name__}: {e}")
    sys.exit(1)

from sklearn.metrics import f1_score


class TabNetTuner:
    """
    Optuna tuner for TabNet hyperparameters.
    
    Optimizes:
    - Architecture: n_d, n_a, n_steps, gamma
    - Training: learning rate, batch size
    
    Objective: Macro-F1 score on validation set
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        class_weights: Dict[int, float],
        n_trials: int = 30,
        pruning_enabled: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize TabNet tuner.
        
        Parameters
        ----------
        X_train, X_val : np.ndarray
            Training and validation features
        y_train, y_val : np.ndarray
            Training and validation targets
        class_weights : Dict[int, float]
            Class weights for imbalance
        n_trials : int
            Number of trials (30-50)
        pruning_enabled : bool
            Enable Optuna pruning for early stopping
        verbose : bool
            Print optimization progress
        """
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.class_weights = class_weights
        self.n_trials = n_trials
        self.pruning_enabled = pruning_enabled
        self.verbose = verbose
        
        # Convert class weights to array
        n_classes = len(class_weights)
        self.class_weights_array = np.array([
            class_weights.get(i, 1.0) for i in range(n_classes)
        ])
        
        # Storage for best results
        self.best_trial = None
        self.best_params = None
        self.best_score = 0.0
        self.history = []
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        
        Returns
        -------
        Dict[str, Any]
            Hyperparameter configuration
        """
        params = {
            # Architecture parameters
            'n_d': trial.suggest_int('n_d', 32, 128, step=16),
            'n_a': trial.suggest_int('n_a', 32, 128, step=16),
            'n_steps': trial.suggest_int('n_steps', 3, 8),
            'gamma': trial.suggest_float('gamma', 1.0, 2.5, step=0.1),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-4, 1e-2, log=True),
            
            # Training parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'batch_size': trial.suggest_int('batch_size', 32, 512, step=32),
            'momentum': trial.suggest_float('momentum', 0.01, 0.1, step=0.01),
        }
        
        return params
    
    def train_and_evaluate(
        self,
        params: Dict[str, Any],
        trial: optuna.Trial = None,
    ) -> float:
        """
        Train TabNet and evaluate on validation set.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Hyperparameter configuration
        trial : optuna.Trial, optional
            Optuna trial for reporting and pruning
        
        Returns
        -------
        float
            Macro-F1 score on validation set
        """
        try:
            # Create model
            model = TabNetClassifier(
                n_d=params['n_d'],
                n_a=params['n_a'],
                n_steps=params['n_steps'],
                gamma=params['gamma'],
                lambda_sparse=params['lambda_sparse'],
                momentum=params['momentum'],
                seed=42,
                optimizer_params={
                    'lr': params['learning_rate'],
                    'weight_decay': 1e-5,
                },
                verbose=0,
            )
            
            # Train with early stopping
            model.fit(
                X_train=self.X_train,
                y_train=self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                eval_metric=['accuracy'],
                max_epochs=100,
                patience=10,
                batch_size=params['batch_size'],
                virtual_batch_size=min(params['batch_size'] // 2, 128),
                num_workers=0,
                weights=self.class_weights,
            )
            
            # Evaluate on validation set
            y_pred = model.predict(self.X_val)
            macro_f1 = f1_score(self.y_val, y_pred, average='macro')
            
            # Report to Optuna for pruning
            if trial is not None and self.pruning_enabled:
                trial.report(macro_f1, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return macro_f1
        
        except Exception as e:
            if self.verbose:
                print(f"    Trial failed: {str(e)}")
            return -1.0  # Return worst score on failure
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        
        Returns
        -------
        float
            Macro-F1 score to maximize
        """
        # Define search space
        params = self.define_search_space(trial)
        
        # Train and evaluate
        score = self.train_and_evaluate(params, trial)
        
        # Track results
        trial_result = {
            'trial_id': trial.number,
            'params': params,
            'score': score,
        }
        self.history.append(trial_result)
        
        if self.verbose:
            print(f"  Trial {trial.number:3d}: F1={score:.4f} | "
                  f"n_d={params['n_d']:3d} n_a={params['n_a']:3d} "
                  f"n_steps={params['n_steps']} "
                  f"lr={params['learning_rate']:.2e}")
        
        return score
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run Optuna optimization.
        
        Returns
        -------
        Dict[str, Any]
            Optimization results including best parameters and score
        """
        if self.verbose:
            print("=" * 60)
            print("Optuna Hyperparameter Tuning for TabNet")
            print("=" * 60)
            print(f"\n[CONFIG] Tuning setup:")
            print(f"  Trials: {self.n_trials} (30-50)")
            print(f"  Pruning: {'enabled' if self.pruning_enabled else 'disabled'}")
            print(f"  Objective: Macro-F1 (validation set)")
            print(f"  Training samples: {self.X_train.shape[0]}")
            print(f"  Validation samples: {self.X_val.shape[0]}")
            print()
        
        # Create sampler and pruner
        sampler = TPESampler(seed=42)
        pruner = MedianPruner() if self.pruning_enabled else optuna.pruners.NopPruner()
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
        )
        
        # Optimize
        if self.verbose:
            print("[OPTIMIZATION] Starting trials...\n")
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )
        
        # Extract results
        self.best_trial = study.best_trial
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Print results
        if self.verbose:
            print(f"\n[RESULTS] Optimization complete!")
            print(f"  Best trial: {self.best_trial.number}")
            print(f"  Best macro-F1: {self.best_score:.4f}")
            print(f"\n[BEST PARAMETERS]")
            for param, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  {param:20s}: {value:.6f}")
                else:
                    print(f"  {param:20s}: {value}")
        
        return {
            'best_trial': self.best_trial.number,
            'best_score': self.best_score,
            'best_params': self.best_params,
            'n_trials': len(study.trials),
            'history': self.history,
        }
    
    def save_results(self, output_path: str = "models/tuning/optuna_results.json"):
        """
        Save tuning results to JSON.
        
        Parameters
        ----------
        output_path : str
            Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            'best_trial': self.best_trial.number,
            'best_score': float(self.best_score),
            'best_params': {
                k: (float(v) if isinstance(v, np.floating) else v)
                for k, v in self.best_params.items()
            },
            'n_trials': len(self.history),
            'history': [
                {
                    'trial_id': h['trial_id'],
                    'score': float(h['score']),
                    'params': {
                        k: (float(v) if isinstance(v, np.floating) else v)
                        for k, v in h['params'].items()
                    }
                }
                for h in self.history
            ],
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.verbose:
            print(f"\n[SAVED] Results: {output_path}")


def main():
    """Main execution - run TabNet hyperparameter tuning"""
    
    try:
        # Load data
        print()
        X_train, X_val, X_test, y_train, y_val, y_test = load_tabnet_data()
        
        print("\n")
        
        # Scale features
        print("[PREPROCESSING] Scaling features...")
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_tabnet_features(
            X_train, X_val, X_test, verbose=False
        )
        print("✓ Features scaled\n")
        
        # Compute class weights
        print("[PREPROCESSING] Computing class weights...")
        class_weights = compute_tabnet_class_weights(y_train, verbose=False)
        print(f"✓ Class weights: {class_weights}\n")
        
        # Run tuning
        tuner = TabNetTuner(
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            y_train=y_train,
            y_val=y_val,
            class_weights=class_weights,
            n_trials=30,  # Start with 30 trials (can increase to 50)
            pruning_enabled=True,
            verbose=True,
        )
        
        results = tuner.optimize()
        
        # Save results
        tuner.save_results("models/tuning/optuna_results.json")
        
        # Print summary
        print("\n" + "=" * 60)
        print("✓ Tuning Complete!")
        print("=" * 60)
        print(f"\nBest Configuration:")
        print(f"  Macro-F1: {results['best_score']:.4f}")
        print(f"  Trials completed: {results['n_trials']}")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
