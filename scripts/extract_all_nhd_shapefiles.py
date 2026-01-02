#!/usr/bin/env python3
"""
Extract all NHD shapefiles from downloaded zip files.

This script extracts NHDFlowline, NHDWaterbody, and NHDArea shapefiles
from all downloaded NHD zip files into a single directory structure.

Usage:
    python scripts/extract_all_nhd_shapefiles.py \
        --zip-dir data/raw/nhd \
        --output-dir data/raw/nhd/shapefiles
"""

import argparse
import sys
import zipfile
from pathlib import Path
import shutil

# Files we want to extract from each zip
SHAPEFILES_TO_EXTRACT = [
    'NHDFlowline.shp',
    'NHDFlowline.shx',
    'NHDFlowline.dbf',
    'NHDFlowline.prj',
    'NHDFlowlineVAA.dbf',  # Value Added Attributes
    'NHDWaterbody.shp',
    'NHDWaterbody.shx',
    'NHDWaterbody.dbf',
    'NHDWaterbody.prj',
    'NHDArea.shp',
    'NHDArea.shx',
    'NHDArea.dbf',
    'NHDArea.prj',
    'NHDSpring.shp',  # Optional
    'NHDSpring.shx',
    'NHDSpring.dbf',
    'NHDSpring.prj',
]


def extract_shapefiles_from_zip(zip_path: Path, output_dir: Path, huc8_code: str) -> dict:
    """
    Extract shapefiles from a single zip file.
    
    Returns:
        dict with counts of extracted files by type
    """
    extracted = {
        'flowline': 0,
        'waterbody': 0,
        'area': 0,
        'spring': 0,
        'other': 0,
    }
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get list of files in zip
            file_list = zf.namelist()
            
            # Extract each shapefile we need
            for file_in_zip in file_list:
                # Check if this is a file we want
                filename = Path(file_in_zip).name
                
                # Skip if not a shapefile component
                if not any(filename.startswith(prefix) for prefix in 
                          ['NHDFlowline', 'NHDWaterbody', 'NHDArea', 'NHDSpring']):
                    continue
                
                # Skip if not a file type we need
                if not any(filename.endswith(ext) for ext in ['.shp', '.shx', '.dbf', '.prj', 'VAA.dbf']):
                    continue
                
                # Create output path
                # Organize by HUC8 code to avoid overwriting
                huc8_dir = output_dir / huc8_code
                huc8_dir.mkdir(parents=True, exist_ok=True)
                
                output_path = huc8_dir / filename
                
                # Extract file
                with zf.open(file_in_zip) as source:
                    with open(output_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                
                # Count by type
                if filename.startswith('NHDFlowline'):
                    extracted['flowline'] += 1
                elif filename.startswith('NHDWaterbody'):
                    extracted['waterbody'] += 1
                elif filename.startswith('NHDArea'):
                    extracted['area'] += 1
                elif filename.startswith('NHDSpring'):
                    extracted['spring'] += 1
                else:
                    extracted['other'] += 1
        
        return extracted
        
    except zipfile.BadZipFile:
        print(f"  ✗ {zip_path.name}: Invalid zip file")
        return extracted
    except Exception as e:
        print(f"  ✗ {zip_path.name}: Error - {e}")
        return extracted


def extract_all_shapefiles(zip_dir: Path, output_dir: Path):
    """Extract shapefiles from all NHD zip files."""
    
    print("="*60)
    print("EXTRACTING NHD SHAPEFILES FROM ZIP FILES")
    print("="*60)
    print()
    
    # Find all zip files
    zip_files = sorted(zip_dir.glob("NHD_H_*_HU8_Shape.zip"))
    
    if not zip_files:
        print(f"ERROR: No NHD zip files found in {zip_dir}")
        return False
    
    print(f"Found {len(zip_files)} zip files to extract")
    print(f"Output directory: {output_dir}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    total_extracted = {
        'flowline': 0,
        'waterbody': 0,
        'area': 0,
        'spring': 0,
        'other': 0,
    }
    
    successful = 0
    failed = 0
    
    # Extract from each zip
    for i, zip_file in enumerate(zip_files, 1):
        # Extract HUC8 code from filename
        # Format: NHD_H_10020001_HU8_Shape.zip
        huc8 = zip_file.stem.replace('NHD_H_', '').replace('_HU8_Shape', '')
        
        print(f"[{i}/{len(zip_files)}] {zip_file.name}")
        
        extracted = extract_shapefiles_from_zip(zip_file, output_dir, huc8)
        
        if sum(extracted.values()) > 0:
            successful += 1
            for key, value in extracted.items():
                total_extracted[key] += value
            print(f"  ✓ Extracted: Flowline={extracted['flowline']}, "
                  f"Waterbody={extracted['waterbody']}, Area={extracted['area']}")
        else:
            failed += 1
            print(f"  ✗ No shapefiles extracted")
    
    # Summary
    print()
    print("="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Successful: {successful}/{len(zip_files)}")
    print(f"Failed: {failed}")
    print()
    print("Files extracted:")
    print(f"  NHDFlowline files: {total_extracted['flowline']}")
    print(f"  NHDWaterbody files: {total_extracted['waterbody']}")
    print(f"  NHDArea files: {total_extracted['area']}")
    print(f"  NHDSpring files: {total_extracted['spring']}")
    print()
    
    # Count unique HUC8 directories
    huc8_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    print(f"Extracted from {len(huc8_dirs)} HUC8 regions")
    print()
    
    if successful > 0:
        print("✓ Extraction complete!")
        print()
        print("Next step:")
        print(f"  python scripts/process_nhd_water_sources.py \\")
        print(f"      --input-dir {output_dir} \\")
        print(f"      --output data/hydrology/water_sources.geojson")
        return True
    else:
        print("✗ No files extracted. Check zip files.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Extract all NHD shapefiles from downloaded zip files'
    )
    parser.add_argument(
        '--zip-dir',
        type=Path,
        default=Path('data/raw/nhd'),
        help='Directory containing NHD zip files (default: data/raw/nhd)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/raw/nhd/shapefiles'),
        help='Directory to extract shapefiles (default: data/raw/nhd/shapefiles)'
    )
    
    args = parser.parse_args()
    
    success = extract_all_shapefiles(args.zip_dir, args.output_dir)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

