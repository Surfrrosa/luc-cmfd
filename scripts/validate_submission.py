"""Validate submission file format for Kaggle competition."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, 'src')
from rle import rle_encode, rle_decode


def validate_submission(csv_path: str, verbose: bool = True) -> bool:
    """
    Validate submission CSV format.

    Args:
        csv_path: Path to submission.csv
        verbose: Print detailed validation info

    Returns:
        True if valid, False otherwise
    """
    if verbose:
        print(f"Validating {csv_path}...")

    # Check file exists
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        return False

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}")
        return False

    # Check columns
    required_cols = {'case_id', 'annotation'}
    if set(df.columns) != required_cols:
        print(f"ERROR: Invalid columns. Expected {required_cols}, got {set(df.columns)}")
        return False

    if verbose:
        print(f"  Columns: OK")

    # Check no NaNs
    if df.isnull().any().any():
        print(f"ERROR: Found NaN values")
        return False

    if verbose:
        print(f"  No NaNs: OK")

    # Check each row
    n_authentic = 0
    n_forged = 0
    errors = []

    for idx, row in df.iterrows():
        case_id = row['case_id']
        annotation = row['annotation']

        # Check case_id is not empty
        if pd.isna(case_id) or str(case_id).strip() == '':
            errors.append(f"Row {idx}: Empty case_id")
            continue

        # Check annotation format
        if annotation == 'authentic':
            n_authentic += 1
        elif isinstance(annotation, str) and annotation.startswith('[') and annotation.endswith(']'):
            # RLE format - try to parse
            try:
                # Remove brackets and parse
                rle_str = annotation[1:-1]
                if rle_str.strip() == '':
                    errors.append(f"Row {idx} ({case_id}): Empty RLE list")
                    continue

                # Parse RLE pairs
                pairs = [int(x.strip()) for x in rle_str.split(',')]

                # Check even number (start, length pairs)
                if len(pairs) % 2 != 0:
                    errors.append(f"Row {idx} ({case_id}): Odd number of RLE values")
                    continue

                # Check all positive
                if any(x <= 0 for x in pairs):
                    errors.append(f"Row {idx} ({case_id}): RLE contains non-positive values")
                    continue

                # Check 1-indexed (starts should be >= 1)
                starts = pairs[::2]
                if any(s < 1 for s in starts):
                    errors.append(f"Row {idx} ({case_id}): RLE starts are 0-indexed (should be 1-indexed)")
                    continue

                n_forged += 1
            except Exception as e:
                errors.append(f"Row {idx} ({case_id}): Invalid RLE format: {e}")
                continue
        else:
            errors.append(f"Row {idx} ({case_id}): Invalid annotation (not 'authentic' or RLE): {annotation}")
            continue

    # Print errors
    if errors:
        print(f"ERROR: Found {len(errors)} validation errors:")
        for err in errors[:10]:  # Show first 10
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False

    if verbose:
        print(f"  Annotations: OK")
        print(f"  Total rows: {len(df)}")
        print(f"  Authentic: {n_authentic}")
        print(f"  Forged: {n_forged}")

    # Check uniqueness of case_ids
    if df['case_id'].duplicated().any():
        dups = df[df['case_id'].duplicated()]['case_id'].tolist()
        print(f"ERROR: Duplicate case_ids found: {dups[:10]}")
        return False

    if verbose:
        print(f"  Unique case_ids: OK")

    # Round-trip test on random samples
    if verbose:
        print(f"\nRound-trip testing...")

    test_masks = [
        np.zeros((100, 100), dtype=np.uint8),  # Empty
        np.ones((100, 100), dtype=np.uint8),   # Full
        np.eye(50, 50, dtype=np.uint8),        # Diagonal
    ]

    for i, mask in enumerate(test_masks):
        # Encode
        rle = rle_encode(mask)

        # Decode
        if rle == 'authentic':
            decoded = np.zeros_like(mask)
        else:
            decoded = rle_decode(rle, mask.shape)

        # Compare
        if not np.array_equal(mask, decoded):
            print(f"ERROR: Round-trip failed for test mask {i}")
            print(f"  Original sum: {mask.sum()}")
            print(f"  Decoded sum: {decoded.sum()}")
            return False

    if verbose:
        print(f"  Round-trip tests: OK")

    # All checks passed
    if verbose:
        print(f"\nValidation PASSED")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate submission CSV')
    parser.add_argument('csv_path', type=str, help='Path to submission.csv')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')

    args = parser.parse_args()

    valid = validate_submission(args.csv_path, verbose=not args.quiet)

    if valid:
        print(f"\nSUCCESS: {args.csv_path} is valid")
        sys.exit(0)
    else:
        print(f"\nFAILED: {args.csv_path} has errors")
        sys.exit(1)


if __name__ == '__main__':
    main()
