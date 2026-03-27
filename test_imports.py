#!/usr/bin/env python
import sys

print("Python version:", sys.version)

try:
    print("\nImporting torch...")
    import torch
    print("[OK] torch imported:", torch.__version__)
except Exception as e:
    print("[FAIL] Failed to import torch:", e)
    sys.exit(1)

try:
    print("\nImporting pytorch_tabnet...")
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("[OK] TabNet imported successfully")
except Exception as e:
    print("[FAIL] Failed to import TabNet:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[OK] All dependencies ready!")
