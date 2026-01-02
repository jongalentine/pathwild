#!/usr/bin/env python3
"""
Create Wyoming state boundary shapefile for MRLC NLCD Viewer upload.

Requirements:
- Must be in .zip file
- Must be in EPSG:3857 (Web Mercator) projection
- Must be singlepart features only

Usage:
    python scripts/create_wyoming_boundary_shapefile.py [--output-file PATH]
"""

import sys
from pathlib import Path
import zipfile
import tempfile
import shutil

try:
    import geopandas as gpd
    from shapely.geometry import box, Polygon
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("ERROR: geopandas and shapely required")
    print("Install with: pip install geopandas shapely")
    print("Or: conda install geopandas shapely")
    sys.exit(1)

try:
    import fiona
    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("WARNING: requests not available - will use bounding box only")


def create_wyoming_boundary():
    """Create Wyoming state boundary shapefile."""
    
    print("=" * 60)
    print("CREATING WYOMING BOUNDARY SHAPEFILE")
    print("=" * 60)
    
    # Wyoming bounding box (WGS84)
    wyoming_bounds = {
        'west': -111.0,
        'east': -104.0,
        'south': 41.0,
        'north': 45.0
    }
    
    print(f"\nWyoming bounds (WGS84):")
    print(f"  Latitude: {wyoming_bounds['south']:.1f}° to {wyoming_bounds['north']:.1f}°")
    print(f"  Longitude: {wyoming_bounds['west']:.1f}° to {wyoming_bounds['east']:.1f}°")
    
    # Try to get actual Wyoming boundary from various sources
    print(f"\nAttempting to download actual Wyoming boundary...")
    
    wyoming_geom = None
    
    # Method 1: Try US Census TIGER/Line (most accurate)
    if HAS_REQUESTS:
        try:
            print(f"  Trying US Census TIGER/Line data...")
            # TIGER/Line 2023 state boundaries
            url = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
            
            # Download and extract
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                zip_path = temp_path / "states.zip"
                
                response = requests.get(url, timeout=30, stream=True)
                if response.status_code == 200:
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Extract and read
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_path)
                    
                    # Find shapefile
                    shp_files = list(temp_path.glob("*.shp"))
                    if shp_files:
                        states = gpd.read_file(shp_files[0])
                        wyoming = states[states['NAME'] == 'Wyoming']
                        if len(wyoming) > 0:
                            # Keep as GeoDataFrame to preserve CRS and structure
                            wyoming_geom = wyoming[['NAME', 'geometry']].copy()
                            print(f"  ✓ Found Wyoming boundary from US Census TIGER/Line")
                            print(f"    CRS: {wyoming_geom.crs}")
        except Exception as e:
            print(f"  ⚠ US Census TIGER/Line failed: {e}")
    else:
        print(f"  ⚠ requests not available - skipping TIGER/Line download")
    
    # Method 2: Use bounding box (reliable fallback)
    if wyoming_geom is None:
        print(f"  Using Wyoming bounding box (covers entire state)")
        # Create rectangle polygon - this will work for MRLC viewer
        wyoming_box = box(
            wyoming_bounds['west'],
            wyoming_bounds['south'],
            wyoming_bounds['east'],
            wyoming_bounds['north']
        )
        wyoming_geom = gpd.GeoSeries([wyoming_box], crs='EPSG:4326')
    elif isinstance(wyoming_geom, gpd.GeoDataFrame) and len(wyoming_geom) == 0:
        print(f"  TIGER/Line data empty, using Wyoming bounding box")
        wyoming_box = box(
            wyoming_bounds['west'],
            wyoming_bounds['south'],
            wyoming_bounds['east'],
            wyoming_bounds['north']
        )
        wyoming_geom = gpd.GeoSeries([wyoming_box], crs='EPSG:4326')
    
    # Create GeoDataFrame
    # Handle both GeoDataFrame and GeoSeries
    if isinstance(wyoming_geom, gpd.GeoDataFrame):
        # Already a GeoDataFrame from TIGER/Line
        gdf = wyoming_geom.copy()
        if 'NAME' not in gdf.columns:
            gdf['NAME'] = 'Wyoming'
        if 'STATE' not in gdf.columns:
            gdf['STATE'] = 'WY'
    elif isinstance(wyoming_geom, gpd.GeoSeries):
        # GeoSeries (bounding box) - create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'NAME': ['Wyoming'],
            'STATE': ['WY'],
            'geometry': wyoming_geom
        })
        # CRS should be set from GeoSeries, but ensure it's set
        if gdf.crs is None:
            gdf.set_crs('EPSG:4326', inplace=True)
    else:
        # Regular geometry list (shouldn't happen, but handle it)
        gdf = gpd.GeoDataFrame({
            'NAME': ['Wyoming'],
            'STATE': ['WY'],
            'geometry': wyoming_geom
        }, crs='EPSG:4326')
    
    # Ensure CRS is set
    if gdf.crs is None:
        gdf.set_crs('EPSG:4326', inplace=True)
    
    # Convert to WGS84 if not already (TIGER/Line might be in different CRS)
    if gdf.crs != 'EPSG:4326':
        print(f"  Converting from {gdf.crs} to WGS84...")
        gdf = gdf.to_crs('EPSG:4326')
    
    print(f"\n--- SHAPEFILE CREATION ---")
    print(f"Original CRS: EPSG:4326 (WGS84)")
    print(f"Geometry type: {gdf.geometry.iloc[0].geom_type}")
    
    # Ensure singlepart (explode multipart if needed)
    print(f"\nEnsuring singlepart features...")
    if gdf.geometry.iloc[0].geom_type == 'MultiPolygon':
        print(f"  Converting MultiPolygon to singlepart...")
        gdf = gdf.explode(index_parts=False)
        gdf = gdf.reset_index(drop=True)
        # Keep only the largest polygon (main state boundary)
        gdf['area'] = gdf.geometry.area
        gdf = gdf.nlargest(1, 'area').drop('area', axis=1)
    
    # Convert to EPSG:3857 (Web Mercator)
    print(f"\nConverting to EPSG:3857 (Web Mercator)...")
    gdf_3857 = gdf.to_crs('EPSG:3857')
    print(f"  ✓ Converted to EPSG:3857")
    
    # Verify singlepart
    for idx, geom in enumerate(gdf_3857.geometry):
        if geom.geom_type == 'MultiPolygon':
            print(f"  ⚠ WARNING: Feature {idx} is still MultiPolygon - converting...")
            gdf_3857 = gdf_3857.explode(index_parts=False)
            gdf_3857 = gdf_3857.reset_index(drop=True)
            # Keep largest polygon
            gdf_3857['area'] = gdf_3857.geometry.area
            gdf_3857 = gdf_3857.nlargest(1, 'area').drop('area', axis=1)
            break
    
    print(f"  Final geometry type: {gdf_3857.geometry.iloc[0].geom_type}")
    print(f"  Final CRS: {gdf_3857.crs}")
    
    # Verify bounds in Web Mercator
    bounds = gdf_3857.total_bounds
    print(f"\nBounds in EPSG:3857:")
    print(f"  X: {bounds[0]:.0f} to {bounds[2]:.0f} meters")
    print(f"  Y: {bounds[1]:.0f} to {bounds[3]:.0f} meters")
    
    return gdf_3857


def create_shapefile_zip(gdf, output_zip_path: Path):
    """Create a zip file containing the shapefile components."""
    
    print(f"\n--- CREATING SHAPEFILE ZIP ---")
    
    # Create temporary directory for shapefile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        shapefile_name = "wyoming_boundary"
        shapefile_path = temp_path / shapefile_name
        
        # Save shapefile
        print(f"  Saving shapefile components...")
        
        # geopandas requires .shp extension in the path
        shapefile_path_str = str(shapefile_path) + '.shp'
        print(f"    Target path: {shapefile_path_str}")
        
        try:
            gdf.to_file(shapefile_path_str, driver='ESRI Shapefile')
            print(f"    ✓ Shapefile saved")
        except Exception as e:
            print(f"    ✗ Failed to save shapefile: {e}")
            print(f"    Error type: {type(e).__name__}")
            # Try alternative method if pyogrio fails
            if HAS_FIONA:
                try:
                    print(f"    Trying alternative save method (fiona)...")
                    # Use fiona directly as fallback
                    from fiona.crs import from_epsg
                    schema = {
                        'geometry': gdf.geometry.iloc[0].geom_type,
                        'properties': {col: 'str' if gdf[col].dtype == 'object' else 'float' 
                                      for col in gdf.columns if col != 'geometry'}
                    }
                    with fiona.open(shapefile_path_str, 'w', driver='ESRI Shapefile', 
                                  crs=from_epsg(3857), schema=schema) as f:
                        for idx, row in gdf.iterrows():
                            props = {k: str(v) if isinstance(v, (list, dict)) else v 
                                    for k, v in row.items() if k != 'geometry'}
                            f.write({
                                'geometry': row.geometry.__geo_interface__,
                                'properties': props
                            })
                    print(f"    ✓ Shapefile saved using alternative method")
                except Exception as e2:
                    print(f"    ✗ Alternative method also failed: {e2}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                print(f"    ✗ fiona not available for fallback method")
                return False
        
        # Get base path (without .shp extension) for finding all files
        base_path = Path(shapefile_path_str).with_suffix('')
        
        # Verify all required files exist
        required_extensions = ['.shp', '.shx', '.dbf', '.prj']
        shapefile_files = []
        
        print(f"\n  Checking for shapefile components...")
        for ext in required_extensions:
            file_path = base_path.with_suffix(ext)
            if file_path.exists():
                shapefile_files.append(file_path)
                size_kb = file_path.stat().st_size / 1024
                print(f"    ✓ {file_path.name} ({size_kb:.1f} KB)")
            else:
                print(f"    ✗ Missing: {file_path.name}")
        
        # If some files are missing, list what actually exists
        if len(shapefile_files) < 4:
            print(f"\n    Files actually found in temp directory:")
            for f in sorted(temp_path.glob("*")):
                size_kb = f.stat().st_size / 1024 if f.is_file() else 0
                print(f"      - {f.name} ({size_kb:.1f} KB)" if f.is_file() else f"      - {f.name}/")
            
            # Try to find files with different naming
            print(f"\n    Searching for shapefile files...")
            for pattern in ['*.shp', '*.shx', '*.dbf', '*.prj']:
                found = list(temp_path.glob(pattern))
                if found:
                    print(f"      Found {pattern}: {[f.name for f in found]}")
                    shapefile_files.extend(found)
            
            # Remove duplicates
            shapefile_files = list(set(shapefile_files))
            
            if len(shapefile_files) < 4:
                return False
        
        if len(shapefile_files) < 4:
            print(f"    ✗ Only {len(shapefile_files)} files found, expected 4")
            return False
        
        # Create zip file
        print(f"\n  Creating zip file: {output_zip_path}")
        try:
            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in shapefile_files:
                    zipf.write(file_path, file_path.name)
                    print(f"    Added: {file_path.name}")
        except Exception as e:
            print(f"    ✗ Failed to create zip: {e}")
            return False
        
        zip_size_mb = output_zip_path.stat().st_size / (1024 * 1024)
        print(f"\n  ✓ Zip file created: {output_zip_path}")
        print(f"    Size: {zip_size_mb:.2f} MB")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create Wyoming boundary shapefile for MRLC upload"
    )
    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path('data/landcover/wyoming_boundary.zip'),
        help='Output zip file path (default: data/landcover/wyoming_boundary.zip)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing shapefile, do not create new one'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        if not args.output_file.exists():
            print(f"✗ File not found: {args.output_file}")
            return 1
        
        print(f"Verifying: {args.output_file}")
        # TODO: Add verification logic
        return 0
    
    # Create output directory
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create Wyoming boundary
    gdf = create_wyoming_boundary()
    
    # Create shapefile zip
    success = create_shapefile_zip(gdf, args.output_file)
    
    if success:
        print(f"\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"\n✓ Wyoming boundary shapefile created: {args.output_file}")
        print(f"\nThis file is ready to upload to MRLC NLCD Viewer!")
        print(f"\nUpload instructions:")
        print(f"  1. Go to: https://www.mrlc.gov/viewer/")
        print(f"  2. Look for 'Upload Shapefile' or 'Import' option")
        print(f"  3. Select this file: {args.output_file}")
        print(f"  4. The viewer should recognize Wyoming boundaries")
        print(f"  5. Proceed with selecting Land Cover product and year")
        return 0
    else:
        print(f"\n✗ Failed to create shapefile zip")
        return 1


if __name__ == "__main__":
    sys.exit(main())

