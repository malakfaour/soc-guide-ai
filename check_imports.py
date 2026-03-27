#!/usr/bin/env python
"""Check import paths quickly"""

import sys

print("\n[Checking imports...]")

try:
    print("  Checking pytorch_tabnet...")
    import pytorch_tabnet
    print(f"    [OK] pytorch_tabnet package loaded")
    print(f"    Package path: {list(pytorch_tabnet.__path__)}")
except Exception as e:
    print(f"    [FAIL] pytorch_tabnet: {type(e).__name__}: {e}")

try:
    print("  Checking torch...")
    import torch
    print(f"    [OK] torch {torch.__version__}")
except Exception as e:
    print(f"    [FAIL] torch: {type(e).__name__}: {e}")

try:
    print("  Checking optuna...")
    import optuna
    print(f"    [OK] optuna {optuna.__version__}")
except Exception as e:
    print(f"    [FAIL] optuna: {type(e).__name__}: {e}")

try:
    print("  Checking sklearn...")
    import sklearn
    print(f"    [OK] sklearn {sklearn.__version__}")
except Exception as e:
    print(f"    [FAIL] sklearn: {type(e).__name__}: {e}")

print("\n[Attempting TabNetClassifier import...]")
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("  [OK] TabNetClassifier imported from pytorch_tabnet.tab_model")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")

print("\nDone!")
