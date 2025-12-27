#!/usr/bin/env python3
"""
Script to download National Elk Refuge and Southern GYE datasets.
These datasets are available from USGS ScienceBase and may require manual download.
This script provides instructions and attempts automated download if possible.
"""

import requests
from pathlib import Path
import json
import zipfile
import os

# Setup directories
DATA_DIR = Path("data/raw")
REFUGE_DIR = DATA_DIR / "elk_national_refuge"
GYE_DIR = DATA_DIR / "elk_southern_gye"

REFUGE_DIR.mkdir(parents=True, exist_ok=True)
GYE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ELK DATASET DOWNLOAD SCRIPT")
print("=" * 70)

# National Elk Refuge dataset
print("\n1. NATIONAL ELK REFUGE DATASET")
print("-" * 70)
print("URL: https://www.sciencebase.gov/catalog/item/5a9f2782e4b0b1c392e502ea")
print("Alternative: https://data.usgs.gov/datacatalog/data/USGS:5a9f2782e4b0b1c392e502ea")

# Try ScienceBase API
refuge_sb_id = "5a9f2782e4b0b1c392e502ea"
refuge_api_url = f"https://www.sciencebase.gov/catalog/item/{refuge_sb_id}?format=json"

try:
    print(f"\nAttempting to access ScienceBase API...")
    response = requests.get(refuge_api_url, timeout=10)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Found dataset: {data.get('title', 'Unknown')}")
        
        # Look for download links
        files = data.get('files', [])
        if files:
            print(f"  Found {len(files)} file(s)")
            for f in files[:3]:  # Show first 3
                print(f"    - {f.get('name', 'Unknown')}")
                url = f.get('url')
                if url:
                    print(f"      URL: {url}")
        else:
            print("  No direct download links found in API response")
    else:
        print(f"  API returned status {response.status_code}")
except Exception as e:
    print(f"  Could not access API: {e}")

# Southern GYE dataset
print("\n2. SOUTHERN GYE DATASET")
print("-" * 70)
print("URL: https://www.sciencebase.gov/catalog/item/5a7b5f0fe4b00f54eb3a6e3d")
print("Alternative: https://catalog.data.gov/dataset/elk-gps-collar-data-in-southern-gye-2007-2015")

gye_sb_id = "5a7b5f0fe4b00f54eb3a6e3d"
gye_api_url = f"https://www.sciencebase.gov/catalog/item/{gye_sb_id}?format=json"

try:
    print(f"\nAttempting to access ScienceBase API...")
    response = requests.get(gye_api_url, timeout=10)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Found dataset: {data.get('title', 'Unknown')}")
        
        # Look for download links
        files = data.get('files', [])
        if files:
            print(f"  Found {len(files)} file(s)")
            for f in files[:3]:  # Show first 3
                print(f"    - {f.get('name', 'Unknown')}")
                url = f.get('url')
                if url:
                    print(f"      URL: {url}")
        else:
            print("  No direct download links found in API response")
    else:
        print(f"  API returned status {response.status_code}")
except Exception as e:
    print(f"  Could not access API: {e}")

print("\n" + "=" * 70)
print("DOWNLOAD INSTRUCTIONS")
print("=" * 70)
print("\nThese datasets typically require manual download from the web interface.")
print("\nFor National Elk Refuge:")
print("  1. Visit: https://www.sciencebase.gov/catalog/item/5a9f2782e4b0b1c392e502ea")
print("  2. Click 'Download' or 'Files' tab")
print("  3. Download the CSV or ZIP file")
print("  4. Extract to: data/raw/elk_national_refuge/")
print("\nFor Southern GYE:")
print("  1. Visit: https://www.sciencebase.gov/catalog/item/5a7b5f0fe4b00f54eb3a6e3d")
print("  2. Click 'Download' or 'Files' tab")
print("  3. Download the CSV or ZIP file")
print("  4. Extract to: data/raw/elk_southern_gye/")
print("\nAfter downloading, run the exploration notebooks:")
print("  - notebooks/03_explore_national_refuge.ipynb")
print("  - notebooks/04_explore_southern_gye.ipynb")
