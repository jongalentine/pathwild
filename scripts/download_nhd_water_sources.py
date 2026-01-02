#!/usr/bin/env python3
"""
Download NHD High Resolution water source data for Wyoming.

This script automates downloading NHD vector data (NHDFlowline, NHDWaterbody, etc.)
for all subregions covering Wyoming.

Usage:
    # Method 1: Automated download (if URLs are available)
    python scripts/download_nhd_water_sources.py --auto
    
    # Method 2: Download from URLs file
    python scripts/download_nhd_water_sources.py --urls-file nhd_urls.txt
    
    # Method 3: Extract URLs from National Map Downloader (requires manual step)
    python scripts/download_nhd_water_sources.py --extract-urls

Requirements:
    pip install requests tqdm
"""

import argparse
import sys
import json
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import zipfile
import re

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Install tqdm for progress bars: pip install tqdm")

# Wyoming NHD subregions (HUC4 codes that cover Wyoming)
# Wyoming spans multiple HUC4 regions
WYOMING_HUC4_REGIONS = [
    '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009',
    '1010', '1011', '1012', '1013', '1014', '1015', '1016', '1017',
    '1018', '1019', '1020', '1021', '1022', '1023', '1024', '1025',
    '1026', '1027', '1028', '1029', '1030', '1031', '1032', '1033',
    '1034', '1035', '1036', '1037', '1038', '1039', '1040', '1041',
]

# Note: Not all of these may be needed. The actual subregions depend on
# NHD High Resolution distribution. The user should verify which ones
# are actually needed from the National Map Downloader.

# Base URL patterns (these may need to be updated based on USGS structure)
# NHD data is typically distributed through the National Map Downloader
# which doesn't have simple direct URLs. However, we can try to construct
# URLs based on known patterns or extract them from the downloader.

NHD_BASE_URLS = {
    # These are example patterns - actual URLs need to be obtained from
    # the National Map Downloader or USGS FTP server
    'ftp': 'ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Hydrography/NHD/HighResolution/GPKG/',
    'https': 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HighResolution/',
}


def download_file(url: str, output_file: Path, max_retries: int = 3) -> tuple:
    """
    Download a file from a URL.
    
    Args:
        url: URL to download
        output_file: Path to save file
        max_retries: Maximum retry attempts
        
    Returns:
        (url, success: bool, error_message: str, size_mb: float)
    """
    # Skip if already downloaded
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        return (url, True, "already exists", size_mb)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=300)  # 5 min timeout
            response.raise_for_status()
            
            # Check file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress
            downloaded = 0
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            size_mb = downloaded / (1024 * 1024)
            
            # Verify file size (should be at least 1 MB for NHD data)
            if size_mb < 1.0:
                output_file.unlink()
                return (url, False, f"File too small ({size_mb:.2f} MB)", 0.0)
            
            return (url, True, f"{size_mb:.1f} MB", size_mb)
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            if output_file.exists():
                output_file.unlink()
            return (url, False, str(e), 0.0)
        except Exception as e:
            if output_file.exists():
                output_file.unlink()
            return (url, False, str(e), 0.0)
    
    return (url, False, "Max retries exceeded", 0.0)


def extract_urls_from_downloader():
    """
    Guide user to extract URLs from National Map Downloader.
    
    This function provides instructions for getting download URLs.
    """
    print("="*60)
    print("EXTRACTING NHD DOWNLOAD URLS")
    print("="*60)
    print()
    print("The National Map Downloader doesn't provide direct API access.")
    print("However, you can extract download URLs using these methods:")
    print()
    print("METHOD 1: Browser Developer Tools")
    print("-" * 60)
    print("1. Go to: https://apps.nationalmap.gov/downloader/")
    print("2. Select 'Data' tab → 'National Hydrography Dataset (NHD)'")
    print("3. Select 'NHD High Resolution'")
    print("4. Select area: Wyoming")
    print("5. Check products: NHDFlowline, NHDWaterbody, NHDArea, NHDSpring")
    print("6. Click 'Find Products'")
    print("7. Open browser Developer Tools (F12)")
    print("8. Go to 'Network' tab")
    print("9. Click 'Download' for each product")
    print("10. Find the download request in Network tab")
    print("11. Copy the request URL")
    print()
    print("METHOD 2: Download Links File")
    print("-" * 60)
    print("1. After selecting products, right-click each download link")
    print("2. 'Copy link address' or 'Copy link location'")
    print("3. Save URLs to a text file (one per line)")
    print("4. Use: python scripts/download_nhd_water_sources.py --urls-file urls.txt")
    print()
    print("METHOD 3: Bulk Download Application")
    print("-" * 60)
    print("1. Go to: https://bulk-cloud.usgs.gov/")
    print("2. Create an account/login")
    print("3. Search for 'NHD High Resolution'")
    print("4. Select Wyoming subregions")
    print("5. Add to cart and download")
    print()
    
    # Ask user if they want to create a template file
    response = input("\nWould you like to create a template URLs file? (y/n): ")
    if response.lower() == 'y':
        template_file = Path('data/raw/nhd/nhd_download_urls.txt')
        template_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(template_file, 'w') as f:
            f.write("# NHD High Resolution Download URLs\n")
            f.write("# Add one URL per line\n")
            f.write("# Example format:\n")
            f.write("# https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HighResolution/GPKG/NHD_H_1002_GPKG.zip\n")
            f.write("# https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HighResolution/GPKG/NHD_H_1003_GPKG.zip\n")
            f.write("# ...\n")
            f.write("\n")
            f.write("# Add your URLs below:\n")
        
        print(f"\n✓ Created template file: {template_file}")
        print("   Add your download URLs to this file, then run:")
        print(f"   python scripts/download_nhd_water_sources.py --urls-file {template_file}")


def load_urls_from_file(urls_file: Path) -> list:
    """Load URLs from a text file."""
    urls = []
    with open(urls_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                # Validate URL
                if line.startswith('http://') or line.startswith('https://') or line.startswith('ftp://'):
                    urls.append(line)
                else:
                    print(f"⚠ Skipping invalid URL: {line}")
    return urls


def download_from_urls(urls: list, output_dir: Path, workers: int = 4):
    """Download files from a list of URLs."""
    print("="*60)
    print("DOWNLOADING NHD DATA")
    print("="*60)
    print(f"URLs to download: {len(urls)}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {workers}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download files
    results = []
    
    def download_with_name(url):
        # Extract filename from URL
        filename = url.split('/')[-1]
        if not filename or '.' not in filename:
            # Generate filename from URL
            filename = f"nhd_{hash(url) % 10000}.zip"
        output_file = output_dir / filename
        return download_file(url, output_file)
    
    start_time = time.time()
    
    # Create progress bar if available
    progress_bar = None
    if HAS_TQDM:
        progress_bar = tqdm(total=len(urls), desc="Downloading", unit="file")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_with_name, url): url for url in urls}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            url, success, message, size_mb = result
            filename = url.split('/')[-1]
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
                if success:
                    progress_bar.set_postfix_str(f"✓ {filename[:30]}")
                else:
                    progress_bar.set_postfix_str(f"✗ {filename[:30]}")
            else:
                if success:
                    print(f"  ✓ {filename}: {message}")
                else:
                    print(f"  ✗ {filename}: {message}")
    
    # Close progress bar
    if progress_bar:
        progress_bar.close()
    
    elapsed = time.time() - start_time
    
    # Summary
    successful = sum(1 for _, success, _, _ in results if success)
    failed = len(results) - successful
    total_size = sum(size_mb for _, _, _, size_mb in results)
    
    print()
    print("="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successful: {successful}/{len(urls)}")
    print(f"Failed: {failed}")
    print(f"Total size: {total_size:.1f} MB")
    print(f"Time: {elapsed:.1f} seconds")
    
    if failed > 0:
        print("\nFailed downloads:")
        for url, success, message, _ in results:
            if not success:
                print(f"  {url.split('/')[-1]}: {message}")
    
    return successful == len(urls)


def extract_shapefiles_from_zips(zip_dir: Path, output_dir: Path):
    """Extract NHDFlowline and NHDWaterbody shapefiles from downloaded zips."""
    print()
    print("="*60)
    print("EXTRACTING SHAPEFILES")
    print("="*60)
    
    zip_files = list(zip_dir.glob("*.zip"))
    if not zip_files:
        print("No zip files found to extract")
        return
    
    print(f"Found {len(zip_files)} zip file(s)")
    
    extracted_count = 0
    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Find NHD shapefiles
                nhd_files = [
                    f for f in zf.namelist()
                    if any(name in f for name in ['NHDFlowline', 'NHDWaterbody', 'NHDArea', 'NHDSpring'])
                    and f.endswith('.shp')
                ]
                
                if nhd_files:
                    print(f"\nExtracting from {zip_file.name}:")
                    for file in nhd_files:
                        # Preserve directory structure
                        output_path = output_dir / Path(file).name
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with zf.open(file) as source, open(output_path, 'wb') as target:
                            target.write(source.read())
                        
                        # Extract associated files (.shx, .dbf, .prj, etc.)
                        base = file[:-4]
                        base_files = [f for f in zf.namelist() if f.startswith(base) and f != file]
                        for base_file in base_files:
                            base_output = output_dir / Path(base_file).name
                            with zf.open(base_file) as source, open(base_output, 'wb') as target:
                                target.write(source.read())
                        
                        print(f"  ✓ {Path(file).name}")
                        extracted_count += 1
        except Exception as e:
            print(f"  ✗ Error extracting {zip_file.name}: {e}")
    
    print(f"\n✓ Extracted {extracted_count} shapefile(s)")


def main():
    parser = argparse.ArgumentParser(
        description='Download NHD High Resolution water source data for Wyoming'
    )
    parser.add_argument(
        '--urls-file',
        type=Path,
        help='Text file with download URLs (one per line)'
    )
    parser.add_argument(
        '--extract-urls',
        action='store_true',
        help='Show instructions for extracting URLs from National Map Downloader'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/raw/nhd'),
        help='Directory to save downloaded files (default: data/raw/nhd)'
    )
    parser.add_argument(
        '--extract-shapefiles',
        action='store_true',
        help='Extract shapefiles from downloaded zip files'
    )
    parser.add_argument(
        '--shapefiles-dir',
        type=Path,
        default=Path('data/raw/nhd/shapefiles'),
        help='Directory to extract shapefiles (default: data/raw/nhd/shapefiles)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel downloads (default: 4)'
    )
    
    args = parser.parse_args()
    
    if args.extract_urls:
        extract_urls_from_downloader()
        return 0
    
    if args.urls_file:
        if not args.urls_file.exists():
            print(f"ERROR: URLs file not found: {args.urls_file}")
            return 1
        
        urls = load_urls_from_file(args.urls_file)
        if not urls:
            print("ERROR: No valid URLs found in file")
            return 1
        
        success = download_from_urls(urls, args.output_dir, args.workers)
        
        if args.extract_shapefiles and success:
            extract_shapefiles_from_zips(args.output_dir, args.shapefiles_dir)
        
        return 0 if success else 1
    else:
        print("="*60)
        print("NHD WATER SOURCES DOWNLOADER")
        print("="*60)
        print()
        print("This script helps automate downloading NHD High Resolution data.")
        print()
        print("OPTIONS:")
        print("1. Extract URLs from National Map Downloader:")
        print("   python scripts/download_nhd_water_sources.py --extract-urls")
        print()
        print("2. Download from URLs file:")
        print("   python scripts/download_nhd_water_sources.py --urls-file urls.txt")
        print()
        print("3. Download and extract shapefiles:")
        print("   python scripts/download_nhd_water_sources.py --urls-file urls.txt --extract-shapefiles")
        print()
        print("For detailed instructions, see:")
        print("  docs/water_sources_integration.md")
        print()
        return 0


if __name__ == '__main__':
    sys.exit(main())

