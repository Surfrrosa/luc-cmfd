#!/usr/bin/env python3
"""
Capture complete environment information for reproducibility.

Usage:
    python scripts/capture_environment.py --output environment.json

Captures:
    - Python version
    - PyTorch version + CUDA
    - All installed packages
    - GPU information
    - System details
"""

import argparse
import json
import sys
import platform
import subprocess
from datetime import datetime


def get_gpu_info():
    """Get GPU information."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                'available': True,
                'count': torch.cuda.device_count(),
                'devices': []
            }

            for i in range(torch.cuda.device_count()):
                gpu_info['devices'].append({
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'compute_capability': f"{torch.cuda.get_device_capability(i)[0]}.{torch.cuda.get_device_capability(i)[1]}",
                    'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / 1e9
                })

            return gpu_info
        else:
            return {'available': False}
    except:
        return {'available': False, 'error': 'Could not detect GPU'}


def get_package_versions():
    """Get all installed package versions."""
    try:
        result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
        packages = {}
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                name, version = line.split('==')
                packages[name] = version
        return packages
    except:
        return {}


def get_key_packages():
    """Get versions of key ML packages."""
    key_packages = {}

    try:
        import torch
        key_packages['torch'] = torch.__version__
        key_packages['cuda_compiled'] = torch.version.cuda if torch.version.cuda else 'N/A'
        key_packages['cudnn'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'
    except:
        pass

    try:
        import torchvision
        key_packages['torchvision'] = torchvision.__version__
    except:
        pass

    try:
        import numpy
        key_packages['numpy'] = numpy.__version__
    except:
        pass

    try:
        import cv2
        key_packages['opencv'] = cv2.__version__
    except:
        pass

    try:
        import sklearn
        key_packages['scikit-learn'] = sklearn.__version__
    except:
        pass

    try:
        import pandas
        key_packages['pandas'] = pandas.__version__
    except:
        pass

    return key_packages


def capture_environment():
    """Capture complete environment information."""

    env_info = {
        'captured_at': datetime.now().isoformat(),

        'python': {
            'version': sys.version,
            'version_info': {
                'major': sys.version_info.major,
                'minor': sys.version_info.minor,
                'micro': sys.version_info.micro
            },
            'executable': sys.executable
        },

        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },

        'gpu': get_gpu_info(),

        'key_packages': get_key_packages(),

        'all_packages': get_package_versions()
    }

    return env_info


def main(args):
    print("Capturing environment information...")

    env_info = capture_environment()

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(env_info, f, indent=2)

    print(f"✓ Environment captured to {args.output}")

    # Print summary
    print("\n=== Environment Summary ===")
    print(f"Python: {env_info['python']['version_info']['major']}.{env_info['python']['version_info']['minor']}.{env_info['python']['version_info']['micro']}")
    print(f"Platform: {env_info['platform']['system']} {env_info['platform']['release']}")

    if env_info['gpu']['available']:
        print(f"GPU: {env_info['gpu']['devices'][0]['name']} ({env_info['gpu']['count']} device(s))")
    else:
        print("GPU: Not available")

    print("\nKey Packages:")
    for pkg, version in env_info['key_packages'].items():
        print(f"  {pkg}: {version}")

    print(f"\nTotal packages: {len(env_info['all_packages'])}")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture environment for reproducibility')
    parser.add_argument('--output', type=str, default='environment.json',
                       help='Output JSON file path')

    args = parser.parse_args()
    sys.exit(main(args))
