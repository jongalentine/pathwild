#!/usr/bin/env python3
"""
Extract Area 048 from the full elk hunt areas shapefile and save as a separate shapefile.
"""

import geopandas as gpd
from pathlib import Path

def extract_area_048():
    """Extract Area 048 from ElkHuntAreas.shp and save as separate shapefile"""
    
    # Input and output paths
    input_shapefile = Path("data/raw/hunt_areas/ElkHuntAreas.shp")
    output_dir = Path("data/raw/hunt_areas")
    output_shapefile = output_dir / "Area_048.shp"
    
    print("=" * 70)
    print("EXTRACTING AREA 048 FROM ELK HUNT AREAS")
    print("=" * 70)
    
    # Load the full shapefile
    print(f"\nLoading: {input_shapefile}")
    gdf_all = gpd.read_file(input_shapefile)
    
    print(f"  Total areas in file: {len(gdf_all)}")
    print(f"  Columns: {list(gdf_all.columns)}")
    
    # Filter for Area 048
    print(f"\nFiltering for Area 048...")
    area_048 = gdf_all[gdf_all['HUNTAREA'] == 48.0].copy()
    
    if len(area_048) == 0:
        print("  ✗ Area 048 not found!")
        return False
    
    print(f"  ✓ Found Area 048!")
    print(f"\nArea 048 Information:")
    print(f"  Hunt Area: {area_048['HUNTAREA'].iloc[0]}")
    print(f"  Hunt Name: {area_048['HUNTNAME'].iloc[0]}")
    print(f"  Herd Unit: {area_048['HERDUNIT'].iloc[0]}")
    print(f"  Herd Name: {area_048['HERDNAME'].iloc[0]}")
    print(f"  Square Miles: {area_048['SqMiles'].iloc[0]:.2f}")
    print(f"  Region: {area_048['Region'].iloc[0]}")
    
    # Get bounds
    bounds = area_048.total_bounds
    print(f"\n  Bounding box:")
    print(f"    Min X (lon): {bounds[0]:.2f}")
    print(f"    Min Y (lat): {bounds[1]:.2f}")
    print(f"    Max X (lon): {bounds[2]:.2f}")
    print(f"    Max Y (lat): {bounds[3]:.2f}")
    
    # Convert to WGS84 to get lat/lon bounds
    if area_048.crs != 'EPSG:4326':
        area_048_wgs84 = area_048.to_crs('EPSG:4326')
        bounds_wgs84 = area_048_wgs84.total_bounds
        print(f"\n  Bounding box (WGS84):")
        print(f"    Min Longitude: {bounds_wgs84[0]:.4f}°")
        print(f"    Min Latitude: {bounds_wgs84[1]:.4f}°")
        print(f"    Max Longitude: {bounds_wgs84[2]:.4f}°")
        print(f"    Max Latitude: {bounds_wgs84[3]:.4f}°")
        
        # Calculate center
        center_lon = (bounds_wgs84[0] + bounds_wgs84[2]) / 2
        center_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2
        print(f"\n  Center point: {center_lat:.4f}°N, {center_lon:.4f}°W")
    
    # Save to new shapefile
    print(f"\nSaving Area 048 shapefile...")
    output_dir.mkdir(parents=True, exist_ok=True)
    area_048.to_file(output_shapefile, driver='ESRI Shapefile')
    
    print(f"  ✓ Saved to: {output_shapefile}")
    
    # Verify the saved file
    print(f"\nVerifying saved shapefile...")
    verify_gdf = gpd.read_file(output_shapefile)
    print(f"  ✓ Verification successful!")
    print(f"    Shape: {verify_gdf.shape}")
    print(f"    CRS: {verify_gdf.crs}")
    
    # Also save as GeoJSON for easier use
    geojson_file = output_dir / "Area_048.geojson"
    area_048.to_file(geojson_file, driver='GeoJSON')
    print(f"  ✓ Also saved as GeoJSON: {geojson_file}")
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  - {output_shapefile}")
    print(f"  - {output_dir / 'Area_048.shx'}")
    print(f"  - {output_dir / 'Area_048.dbf'}")
    print(f"  - {output_dir / 'Area_048.prj'}")
    print(f"  - {geojson_file}")
    
    return True

if __name__ == "__main__":
    extract_area_048()
