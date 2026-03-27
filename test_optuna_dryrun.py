#!/usr/bin/env python
"""
Dry-run test for Optuna TabNet tuner
Verifies data loading, scaling, weighting, and single trial execution
without running full optimization (safe to run on CPU)
"""

import sys
import json
from pathlib import Path

print("\n" + "=" * 70)
print("OPTUNA TABNET TUNER - DRY RUN TEST")
print("=" * 70)

print("\n[STEP 1] Importing dependencies...")
try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score
    from pytorch_tabnet.tab_model import TabNetClassifier
    
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    print("  ✓ All dependencies available")
except ImportError as e:
    print(f"  ✗ Missing dependency: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[STEP 2] Loading data...")
try:
    sys.path.insert(0, str(Path.cwd()))
    from src.training.train_tabnet import load_tabnet_data
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_tabnet_data()
    
    print(f"  ✓ Data loaded successfully")
    print(f"      Train: X {X_train.shape} y {y_train.shape}")
    print(f"      Val:   X {X_val.shape} y {y_val.shape}")
    print(f"      Test:  X {X_test.shape} y {y_test.shape}")
except Exception as e:
    print(f"  ✗ Data loading failed: {e}")
    sys.exit(1)

print("\n[STEP 3] Computing class weights...")
try:
    from src.models.tabnet.utils import compute_tabnet_class_weights
    
    class_weights = compute_tabnet_class_weights(y_train)
    print(f"  ✓ Class weights computed")
    for cls, weight in class_weights.items():
        print(f"      Class {cls}: {weight:.4f}")
except Exception as e:
    print(f"  ✗ Class weight computation failed: {e}")
    sys.exit(1)

print("\n[STEP 4] Scaling features...")
try:
    from src.models.tabnet.utils import scale_tabnet_features
    
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_tabnet_features(
        X_train, X_val, X_test
    )
    
    print(f"  ✓ Features scaled using QuantileTransformer")
    print(f"      Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    print(f"      Val mean:   {X_val_scaled.mean():.4f}, std: {X_val_scaled.std():.4f}")
except Exception as e:
    print(f"  ✗ Scaling failed: {e}")
    sys.exit(1)

print("\n[STEP 5] Testing TabNetTuner instantiation...")
try:
    from src.tuning.optuna_tabnet import TabNetTuner
    
    tuner = TabNetTuner(
        X_train_scaled, X_val_scaled, y_train, y_val,
        class_weights, 
        n_trials=1,  # Just one trial for dry-run
        pruning_enabled=True,
        verbose=False,
    )
    
    print(f"  ✓ TabNetTuner instantiated successfully")
    print(f"      Search space: {tuner.search_space_size} configurations")
    print(f"      Max trials: {tuner.n_trials}")
except Exception as e:
    print(f"  ✗ TabNetTuner instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[STEP 6] Running single trial (smoke test)...")
try:
    print("  Running trial 1/1... (this may take 2-3 minutes on CPU)")
    
    study = optuna.create_study(
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(),
        direction='maximize',
    )
    
    # Run just one trial
    study.optimize(tuner.objective, n_trials=1, show_progress_bar=True)
    
    best_trial = study.best_trial
    best_f1 = best_trial.value
    
    print(f"\n  ✓ Single trial completed")
    print(f"      Best F1 (validation): {best_f1:.4f}")
    print(f"      Trial parameters:")
    for key, val in best_trial.params.items():
        print(f"          {key}: {val}")
    
except Exception as e:
    print(f"  ✗ Single trial failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ DRY RUN SUCCESSFUL - Ready for full tuning!")
print("=" * 70)

print("""
Next steps:
  1. Run full tuning (30 trials):
     python src/tuning/optuna_tabnet.py
     
  2. Or customize trial count:
     from src.tuning.optuna_tabnet import main
     main(n_trials=50)  # 50 trials instead of 30
""")

print("\n✓ Ready to proceed with hyperparameter optimization!")
