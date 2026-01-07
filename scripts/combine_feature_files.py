#!/usr/bin/env python3
"""
Combine all dataset feature files into a single complete context file.

This script combines all feature CSV files in data/features/ into one
complete_context.csv file for model training.

Usage:
    python scripts/combine_feature_files.py [features_dir] [output_file]
    python scripts/combine_feature_files.py
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
from typing import List, Optional


def combine_feature_files(
    features_dir: Path,
    output_file: Path,
    exclude_test_files: bool = True,
    test_mode: bool = False
) -> pd.DataFrame:
    """
    Combine all feature CSV files into a single DataFrame.
    
    Args:
        features_dir: Directory containing feature CSV files
        output_file: Path to output combined CSV file
        exclude_test_files: If True, exclude files with '_test' in name (when not in test_mode)
        test_mode: If True, only look for and combine test files (*_features_test.csv)
    
    Returns:
        Combined DataFrame
    """
    features_dir = Path(features_dir)
    
    # Find all feature files
    if test_mode:
        # In test mode, only look for test files
        feature_files = list(features_dir.glob('*_features_test.csv'))
        print(f"⚠️  TEST MODE: Only combining test feature files")
    elif exclude_test_files:
        # Default: exclude test files
        feature_files = list(features_dir.glob('*_features.csv'))
        # Double-check to exclude test files
        feature_files = [f for f in feature_files if '_test' not in f.stem]
    else:
        # Include all feature files (regular and test)
        feature_files = list(features_dir.glob('*_features*.csv'))
    
    if not feature_files:
        pattern = "*_features_test.csv" if test_mode else "*_features.csv"
        raise ValueError(
            f"No feature files found in {features_dir}. "
            f"Expected files matching pattern: {pattern}"
        )
    
    print(f"Found {len(feature_files)} feature file(s) to combine:")
    for f in sorted(feature_files):
        print(f"  - {f.name}")
    
    # Load and combine all files
    dfs: List[pd.DataFrame] = []
    for file_path in sorted(feature_files):
        print(f"\nLoading: {file_path.name}")
        df = pd.read_csv(file_path)
        print(f"  Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        dfs.append(df)
    
    # Combine all DataFrames
    print(f"\n{'='*70}")
    print("Combining datasets...")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns")
    
    # Verify target column exists
    if 'elk_present' not in combined_df.columns:
        raise ValueError("Target column 'elk_present' not found in combined dataset")
    
    # Report target distribution
    target_counts = combined_df['elk_present'].value_counts().sort_index()
    print(f"\nTarget distribution (elk_present):")
    for value, count in target_counts.items():
        pct = (count / len(combined_df)) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
    
    # Check for column alignment issues
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)
    
    missing_cols = all_columns - set(combined_df.columns)
    if missing_cols:
        print(f"\n⚠️  Warning: Some columns were missing in some datasets:")
        for col in sorted(missing_cols):
            print(f"    - {col}")
    
    # Save combined file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"Saved combined dataset to: {output_file}")
    print(f"Final shape: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns")
    
    return combined_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Combine all dataset feature files into a single complete context file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine all feature files (default)
  python scripts/combine_feature_files.py
  
  # Test mode: combine test files only
  python scripts/combine_feature_files.py --test
  python scripts/combine_feature_files.py --limit 1000
  
  # Specify custom directories
  python scripts/combine_feature_files.py \\
      --features-dir data/features \\
      --output-file data/features/complete_context.csv
  
  # Include test files (combine both regular and test files)
  python scripts/combine_feature_files.py --include-test-files
        """
    )
    parser.add_argument(
        '--features-dir',
        type=Path,
        default=Path('data/features'),
        help='Directory containing feature CSV files (default: data/features)'
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('data/features/complete_context.csv'),
        help='Output path for combined CSV file (default: data/features/complete_context.csv)'
    )
    parser.add_argument(
        '--include-test-files',
        action='store_true',
        help='Include test files (files with _test in name)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Test mode: combine only test feature files (*_features_test.csv) and output to complete_context_test.csv'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: combine only test feature files (*_features_test.csv) and output to complete_context_test.csv'
    )
    
    args = parser.parse_args()
    
    # Determine test mode
    test_mode = args.test or (args.limit is not None)
    
    # If test mode and output file not specified, use test output file
    if test_mode and args.output_file == Path('data/features/complete_context.csv'):
        # Use default test output file
        output_file = args.features_dir / "complete_context_test.csv"
    else:
        output_file = args.output_file
    
    try:
        combine_feature_files(
            features_dir=args.features_dir,
            output_file=output_file,
            exclude_test_files=not args.include_test_files,
            test_mode=test_mode
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

