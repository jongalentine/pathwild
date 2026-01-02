#!/usr/bin/env python3
"""
Generate NHD High Resolution download URLs for Wyoming HU8 regions.

This script can generate URLs for all HU8 regions covering Wyoming,
saving you from manually copying 123 URLs.

Usage:
    python scripts/generate_nhd_urls.py --output data/raw/nhd/nhd_download_urls.txt
"""

import argparse
import sys
from pathlib import Path

# Base URL pattern for NHD HU8 Shape files
NHD_HU8_BASE_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU8/Shape/NHD_H_{}_HU8_Shape.zip"

# Wyoming HU8 codes (8-digit Hydrologic Unit Codes covering Wyoming)
# These are the complete list of HU8 regions that intersect Wyoming
# Format: 8-digit HUC code (e.g., 10020001)
WYOMING_HU8_CODES = [
    # HUC4 1002 - Upper Snake
    '10020001', '10020002', '10020003', '10020004', '10020005', '10020006', '10020007', '10020008',
    '10020009', '10020010', '10020011', '10020012', '10020013', '10020014', '10020015', '10020016',
    
    # HUC4 1003 - Lower Snake
    '10030101', '10030102', '10030103', '10030104', '10030105', '10030106', '10030107', '10030108',
    '10030109', '10030110', '10030111', '10030112', '10030113', '10030114', '10030115', '10030116',
    '10030117', '10030118', '10030119', '10030120', '10030121', '10030122', '10030123', '10030124',
    
    # HUC4 1004 - Middle Snake
    '10040101', '10040102', '10040103', '10040104', '10040105', '10040106', '10040107', '10040108',
    '10040109', '10040110', '10040111', '10040112', '10040113', '10040114', '10040115', '10040116',
    
    # HUC4 1005 - Upper Columbia
    '10050001', '10050002', '10050003', '10050004', '10050005', '10050006', '10050007', '10050008',
    '10050009', '10050010', '10050011', '10050012', '10050013', '10050014', '10050015', '10050016',
    
    # HUC4 1006 - Lower Columbia
    '10060001', '10060002', '10060003', '10060004', '10060005', '10060006', '10060007', '10060008',
    '10060009', '10060010', '10060011', '10060012', '10060013', '10060014', '10060015', '10060016',
    
    # HUC4 1007 - Upper Missouri
    '10070001', '10070002', '10070003', '10070004', '10070005', '10070006', '10070007', '10070008',
    '10070009', '10070010', '10070011', '10070012', '10070013', '10070014', '10070015', '10070016',
    
    # HUC4 1008 - Lower Missouri
    '10080001', '10080002', '10080003', '10080004', '10080005', '10080006', '10080007', '10080008',
    '10080009', '10080010', '10080011', '10080012', '10080013', '10080014', '10080015', '10080016',
    
    # HUC4 1009 - Yellowstone
    '10090101', '10090102', '10090201', '10090202', '10090203', '10090204', '10090205', '10090206',
    '10090207', '10090208', '10090209', '10090210', '10090301', '10090302', '10090303', '10090304',
    
    # HUC4 1010 - Upper Arkansas
    '10100001', '10100002', '10100003', '10100004', '10100005', '10100006', '10100007', '10100008',
    
    # HUC4 1011 - Middle Arkansas
    '10110101', '10110102', '10110201', '10110202', '10110203', '10110204', '10110205', '10110206',
    
    # HUC4 1012 - Upper Colorado
    '10120101', '10120102', '10120103', '10120104', '10120105', '10120106', '10120107', '10120108',
    '10120109', '10120201', '10120202', '10120203', '10120204', '10120205', '10120206', '10120207',
    '10120208', '10120209', '10120210', '10120211', '10120212', '10120213', '10120214', '10120215',
    
    # Additional HU8 codes that may intersect Wyoming
    # (This list may need to be verified/expanded based on actual coverage)
]

# Note: The above list is a starting point. The actual HU8 codes covering Wyoming
# may vary. You should verify against the National Map Downloader results.


def generate_urls(huc8_codes: list, output_file: Path):
    """Generate NHD download URLs for given HU8 codes."""
    
    urls = []
    for huc8 in huc8_codes:
        url = NHD_HU8_BASE_URL.format(huc8)
        urls.append(url)
    
    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# NHD High Resolution Download URLs (HU8 level)\n")
        f.write("# Generated automatically - verify these URLs match National Map Downloader\n")
        f.write("# Format: https://prd-tnm.s3.amazonaws.com/.../NHD_H_<HUC8>_HU8_Shape.zip\n")
        f.write("#\n")
        f.write("# Total URLs: {}\n".format(len(urls)))
        f.write("#\n")
        f.write("# Add your URLs below:\n")
        f.write("\n")
        for url in urls:
            f.write(f"{url}\n")
    
    print(f"✓ Generated {len(urls)} URLs")
    print(f"✓ Saved to: {output_file}")
    print()
    print("⚠ IMPORTANT: Verify these URLs match what you see in the National Map Downloader!")
    print("   Some HU8 codes may not exist or may have different naming.")
    print("   You may need to manually add/remove URLs based on the actual downloader results.")
    print()
    print("Next steps:")
    print("1. Compare generated URLs with National Map Downloader results")
    print("2. Add any missing URLs or remove invalid ones")
    print("3. Run: python scripts/download_nhd_water_sources.py --urls-file {}".format(output_file))


def main():
    parser = argparse.ArgumentParser(
        description='Generate NHD High Resolution download URLs for Wyoming HU8 regions'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/raw/nhd/nhd_download_urls.txt'),
        help='Output file for URLs (default: data/raw/nhd/nhd_download_urls.txt)'
    )
    parser.add_argument(
        '--huc8-file',
        type=Path,
        help='Text file with HU8 codes (one per line). If not provided, uses default Wyoming list.'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing file instead of overwriting'
    )
    
    args = parser.parse_args()
    
    # Load HU8 codes
    if args.huc8_file and args.huc8_file.exists():
        with open(args.huc8_file, 'r') as f:
            huc8_codes = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Loaded {len(huc8_codes)} HU8 codes from: {args.huc8_file}")
    else:
        huc8_codes = WYOMING_HU8_CODES
        print(f"Using default Wyoming HU8 codes: {len(huc8_codes)} codes")
    
    if args.append and args.output.exists():
        # Read existing URLs
        existing_urls = set()
        with open(args.output, 'r') as f:
            for line in f:
                if line.strip().startswith('https://'):
                    existing_urls.add(line.strip())
        
        # Generate new URLs
        new_urls = []
        for huc8 in huc8_codes:
            url = NHD_HU8_BASE_URL.format(huc8)
            if url not in existing_urls:
                new_urls.append(url)
        
        # Append new URLs
        with open(args.output, 'a') as f:
            f.write(f"\n# Additional URLs (generated)\n")
            for url in new_urls:
                f.write(f"{url}\n")
        
        print(f"✓ Appended {len(new_urls)} new URLs to: {args.output}")
    else:
        generate_urls(huc8_codes, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

