#!/usr/bin/env python3
"""
Setup a new experiment directory with proper structure.

Usage:
    python scripts/setup_experiment.py --name M1-baseline
    python scripts/setup_experiment.py --name M2-ensemble --parent runs/week2/

Creates:
    runs/YYYY-MM-DD_HH-MM-SS_experiment-name/
        ├── config.yaml          # Copy of config used
        ├── train_log.csv        # Per-epoch metrics
        ├── best_model.pth       # Best checkpoint
        ├── val_predictions.npz  # Validation predictions
        ├── sanity_check.png     # Visualization
        └── metadata.json        # Git hash, command, etc.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


def get_git_info():
    """Get current git commit hash and branch."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()

        # Check for uncommitted changes
        status = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
        dirty = bool(status)

        return {
            'commit': commit,
            'branch': branch,
            'dirty': dirty,
            'status': status if dirty else None
        }
    except:
        return None


def get_environment_info():
    """Get key environment information."""
    import sys
    import platform

    env = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'system': platform.system()
    }

    try:
        import torch
        env['torch_version'] = torch.__version__
        env['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env['cuda_version'] = torch.version.cuda
            env['gpu_name'] = torch.cuda.get_device_name(0)
    except:
        pass

    return env


def setup_experiment(name, parent_dir='runs', config_path=None, notes=None):
    """Create experiment directory with metadata."""

    # Create timestamped directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_name = f"{timestamp}_{name}"
    exp_dir = Path(parent_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Created experiment directory: {exp_dir}")

    # Copy config if provided
    if config_path and Path(config_path).exists():
        shutil.copy(config_path, exp_dir / 'config.yaml')
        print(f"✓ Copied config from {config_path}")

    # Create metadata
    metadata = {
        'name': name,
        'created': timestamp,
        'git': get_git_info(),
        'environment': get_environment_info(),
        'notes': notes,
        'config_path': str(config_path) if config_path else None,
    }

    with open(exp_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata with git hash and environment info")

    # Create empty log file with header
    with open(exp_dir / 'train_log.csv', 'w') as f:
        f.write('epoch,train_loss,train_f1,val_loss,val_f1,lr,best\n')
    print(f"✓ Created train_log.csv")

    # Create README with usage instructions
    readme = f"""# Experiment: {name}

Created: {timestamp}

## Directory Structure
- `config.yaml` - Training configuration
- `train_log.csv` - Per-epoch metrics
- `best_model.pth` - Best checkpoint (by val F1)
- `val_predictions.npz` - Validation predictions for post-processing sweep
- `sanity_check.png` - Visual sanity check of predictions
- `metadata.json` - Git info, command, notes

## Training Command
```bash
python src/train.py \\
    --config {exp_dir}/config.yaml \\
    --output_dir {exp_dir} \\
    --log_csv {exp_dir}/train_log.csv \\
    --save_val_preds {exp_dir}/val_predictions.npz
```

## Post-Processing Sweep
```bash
python scripts/sweep_threshold.py \\
    --val_preds {exp_dir}/val_predictions.npz \\
    --output {exp_dir}/threshold_sweep.csv
```

## Create Submission
```bash
python scripts/create_submission.py \\
    --weights {exp_dir}/best_model.pth \\
    --config {exp_dir}/config.yaml \\
    --test_dir /path/to/test_images \\
    --output {exp_dir}/submission.csv
```

## Visualize
```bash
python scripts/visualize_predictions.py \\
    --val_preds {exp_dir}/val_predictions.npz \\
    --output {exp_dir}/sanity_check.png
```
"""

    with open(exp_dir / 'README.md', 'w') as f:
        f.write(readme)
    print(f"✓ Created README.md")

    print(f"\n{'='*60}")
    print(f"Experiment directory ready: {exp_dir}")
    print(f"{'='*60}\n")

    return exp_dir


def main(args):
    exp_dir = setup_experiment(
        name=args.name,
        parent_dir=args.parent,
        config_path=args.config,
        notes=args.notes
    )

    # Print example training command
    print("Next step: Start training with:")
    print(f"\n  python src/train.py \\")
    print(f"      --config {exp_dir}/config.yaml \\")
    print(f"      --output_dir {exp_dir} \\")
    print(f"      --log_csv {exp_dir}/train_log.csv \\")
    print(f"      --save_val_preds {exp_dir}/val_predictions.npz\n")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup new experiment directory')
    parser.add_argument('--name', type=str, required=True,
                       help='Experiment name (e.g., M1-baseline)')
    parser.add_argument('--parent', type=str, default='runs',
                       help='Parent directory for runs (default: runs)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file to copy (optional)')
    parser.add_argument('--notes', type=str, default=None,
                       help='Experiment notes (optional)')

    args = parser.parse_args()
    exit(main(args))
