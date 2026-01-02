#!/usr/bin/env python3
"""
Validate NHD download URLs to check which ones actually exist.

This script checks each URL to see if it returns a 200 (exists) or 404 (doesn't exist).
This helps filter out URLs that were generated but don't have actual data.

Usage:
    python scripts/validate_nhd_urls.py \
        --urls-file data/raw/nhd/nhd_download_urls.txt \
        --output data/raw/nhd/nhd_download_urls_validated.txt
"""

import argparse
import sys
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Install tqdm for progress bars: pip install tqdm")


def check_url_exists(url: str, timeout: int = 10) -> tuple:
    """
    Check if a URL exists by making a HEAD request, fallback to GET if needed.
    
    Returns:
        (url, exists: bool, status_code: int)
    """
    try:
        # Try HEAD first (faster, doesn't download)
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            return (url, True, response.status_code)
        # If HEAD returns 403 (some S3 buckets block HEAD), try GET with range
        elif response.status_code == 403:
            # Try GET with range to only get first byte
            get_response = requests.get(url, timeout=timeout, allow_redirects=True, 
                                        headers={'Range': 'bytes=0-0'}, stream=True)
            exists = get_response.status_code in [200, 206]  # 206 = partial content
            return (url, exists, get_response.status_code)
        else:
            return (url, False, response.status_code)
    except requests.exceptions.RequestException:
        # If HEAD fails, try GET with range as fallback
        try:
            get_response = requests.get(url, timeout=timeout, allow_redirects=True,
                                       headers={'Range': 'bytes=0-0'}, stream=True)
            exists = get_response.status_code in [200, 206]
            return (url, exists, get_response.status_code)
        except:
            return (url, False, 0)


def validate_urls(urls_file: Path, output_file: Path, workers: int = 10):
    """Validate URLs and save only the ones that exist."""
    
    print("="*60)
    print("VALIDATING NHD DOWNLOAD URLS")
    print("="*60)
    
    # Load URLs
    urls = []
    with open(urls_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and line.startswith('http'):
                urls.append(line)
    
    print(f"Total URLs to check: {len(urls)}")
    print(f"Checking with {workers} parallel workers...")
    print()
    
    # Check URLs
    valid_urls = []
    invalid_urls = []
    
    iterator = urls
    if HAS_TQDM:
        iterator = tqdm(urls, desc="Validating")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_url_exists, url): url for url in urls}
        
        for future in as_completed(futures):
            url, exists, status_code = future.result()
            
            if exists:
                valid_urls.append(url)
                if HAS_TQDM:
                    iterator.set_postfix_str(f"✓ {len(valid_urls)} valid")
                else:
                    print(f"  ✓ {url.split('/')[-1]}")
            else:
                invalid_urls.append((url, status_code))
                if not HAS_TQDM:
                    print(f"  ✗ {url.split('/')[-1]} (status: {status_code})")
    
    elapsed = time.time() - start_time
    
    # Save valid URLs
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# NHD High Resolution Download URLs (Validated)\n")
        f.write("# Only URLs that actually exist (returned 200 status)\n")
        f.write(f"# Validated from: {urls_file}\n")
        f.write(f"# Valid URLs: {len(valid_urls)}/{len(urls)}\n")
        f.write("#\n")
        f.write("# Add your URLs below:\n")
        f.write("\n")
        for url in valid_urls:
            f.write(f"{url}\n")
    
    # Summary
    print()
    print("="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Valid URLs: {len(valid_urls)}")
    print(f"Invalid URLs: {len(invalid_urls)}")
    print(f"Total checked: {len(urls)}")
    print(f"Time: {elapsed:.1f} seconds")
    print()
    print(f"✓ Saved valid URLs to: {output_file}")
    
    if invalid_urls:
        print(f"\nInvalid URLs (first 10):")
        for url, status in invalid_urls[:10]:
            print(f"  {url.split('/')[-1]}: status {status}")
        if len(invalid_urls) > 10:
            print(f"  ... and {len(invalid_urls) - 10} more")
    
    return len(valid_urls)


def main():
    parser = argparse.ArgumentParser(
        description='Validate NHD download URLs to find which ones actually exist'
    )
    parser.add_argument(
        '--urls-file',
        type=Path,
        default=Path('data/raw/nhd/nhd_download_urls.txt'),
        help='Input file with URLs to validate'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/raw/nhd/nhd_download_urls_validated.txt'),
        help='Output file for validated URLs'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers (default: 10)'
    )
    
    args = parser.parse_args()
    
    if not args.urls_file.exists():
        print(f"ERROR: URLs file not found: {args.urls_file}")
        return 1
    
    valid_count = validate_urls(args.urls_file, args.output, args.workers)
    
    print()
    print("Next step:")
    print(f"  python scripts/download_nhd_water_sources.py --urls-file {args.output}")
    
    return 0 if valid_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())

