## Dataset Versioning

All team members must use the **exact same dataset version** for reproducibility.

### Current Version
**v1** - Available at `data/processed/v1/`

### Loading Dataset (for all team members)

```python
from src.utils.versioning import load_dataset_by_version

# Load v1
X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_by_version('v1')
```

### Available Commands

```python
from src.utils.versioning import list_versions, get_current_version

# See all versions
versions = list_versions()

# Get current version
current_version = get_current_version()
```

### Creating New Versions
Only when preprocessing pipeline is re-run with different parameters:

```bash
python src/training/test_pipeline.py
```

This will:
- Detect current version
- Create next version (v1 → v2)
- Save all dataset files
- Print: "Dataset version v2 created successfully"

### Protection
- ✅ Existing versions are **protected from overwrite**
- ✅ Manual confirmation required for version changes
- ✅ All team members use same version

See [docs/VERSIONING.md](docs/VERSIONING.md) for complete documentation.
