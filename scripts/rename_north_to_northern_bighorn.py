#!/usr/bin/env python3
"""
Rename files from 'north_bighorn' to 'northern_bighorn' for standardization.

This script safely renames all processed and feature files to match the
standardized dataset name.

Usage:
    python scripts/rename_north_to_northern_bighorn.py [--dry-run]
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import sys


def find_files_to_rename(base_dir: Path, pattern: str) -> List[Path]:
    """Find all files matching the pattern."""
    files = []
    for file_path in base_dir.rglob(pattern):
        if file_path.is_file():
            files.append(file_path)
    return sorted(files)


def rename_file(old_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Rename a file from north_bighorn to northern_bighorn.
    
    Returns:
        (success: bool, message: str)
    """
    new_path = Path(str(old_path).replace('north_bighorn', 'northern_bighorn'))
    
    # Check if target already exists
    if new_path.exists():
        # Check if they're the same file (already renamed)
        if old_path.samefile(new_path):
            return False, f"Already using correct name: {old_path}"
        
        # Target exists and is different - ask user or skip
        return False, f"Target exists: {new_path} (skipping {old_path})"
    
    # Check if old file exists
    if not old_path.exists():
        return False, f"Source file not found: {old_path}"
    
    if dry_run:
        return True, f"Would rename: {old_path} -> {new_path}"
    
    try:
        # Ensure parent directory exists
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rename the file
        old_path.rename(new_path)
        return True, f"Renamed: {old_path} -> {new_path}"
    except Exception as e:
        return False, f"Error renaming {old_path}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Rename files from 'north_bighorn' to 'northern_bighorn'"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be renamed without actually renaming'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Base data directory (default: data)'
    )
    
    args = parser.parse_args()
    data_dir = args.data_dir
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    # Find all files with 'north_bighorn' in the name
    processed_dir = data_dir / 'processed'
    features_dir = data_dir / 'features'
    
    files_to_rename = []
    
    # Processed files
    if processed_dir.exists():
        files_to_rename.extend(find_files_to_rename(processed_dir, '*north_bighorn*'))
    
    # Feature files
    if features_dir.exists():
        files_to_rename.extend(find_files_to_rename(features_dir, '*north_bighorn*'))
    
    if not files_to_rename:
        print("No files found matching 'north_bighorn' pattern.")
        return 0
    
    print(f"Found {len(files_to_rename)} file(s) to rename:")
    print()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be renamed")
        print()
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for old_path in files_to_rename:
        success, message = rename_file(old_path, dry_run=args.dry_run)
        print(message)
        
        if success:
            success_count += 1
        elif "Already using correct name" in message or "Target exists" in message:
            skip_count += 1
        else:
            error_count += 1
    
    print()
    print("=" * 70)
    if args.dry_run:
        print("DRY RUN SUMMARY")
    else:
        print("RENAME SUMMARY")
    print("=" * 70)
    print(f"Successfully renamed: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    
    # Special case: Check for duplicate points files
    north_points = processed_dir / 'north_bighorn_points.csv'
    northern_points = processed_dir / 'northern_bighorn_points.csv'
    
    if north_points.exists() and northern_points.exists():
        print()
        print("=" * 70)
        print("DUPLICATE POINTS FILES DETECTED")
        print("=" * 70)
        print(f"Found both:")
        print(f"  - {north_points} ({north_points.stat().st_size} bytes, modified {north_points.stat().st_mtime})")
        print(f"  - {northern_points} ({northern_points.stat().st_size} bytes, modified {northern_points.stat().st_mtime})")
        print()
        if northern_points.stat().st_mtime > north_points.stat().st_mtime:
            print(f"⚠️  {northern_points} is newer. Consider removing {north_points} manually.")
        else:
            print(f"⚠️  {north_points} is newer. Consider which one to keep.")
        print()
    
    if error_count > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

