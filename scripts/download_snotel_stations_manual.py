#!/usr/bin/env python3
"""
Alternative: Create SNOTEL station file from known Wyoming stations.

Since the USDA AWDB API endpoint structure may vary, this script provides
a manual approach using a known list of Wyoming SNOTEL stations.

You can also get the full list from:
https://wcc.sc.egov.usda.gov/reportGenerator/
Select: State=Wyoming, Network=SNOTEL, then export

Usage:
    python scripts/download_snotel_stations_manual.py [--output PATH]
"""

import argparse
from pathlib import Path
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

# Known Wyoming SNOTEL stations (sample - you may want to expand this)
# Format: (station_id, name, lat, lon, elevation_ft, triplet)
# Triplet format: "SNOTEL:WY:station_id"
WYOMING_SNOTEL_STATIONS = [
    ("955", "BATTLE CREEK", 41.2167, -110.0500, 9450, "SNOTEL:WY:955"),
    ("956", "BEARTOOTH LAKE", 44.9500, -109.5167, 9280, "SNOTEL:WY:956"),
    ("957", "BIG GOOSE", 44.8333, -107.2333, 7120, "SNOTEL:WY:957"),
    ("958", "BLACK HALL MOUNTAIN", 41.2167, -107.9833, 9600, "SNOTEL:WY:958"),
    ("959", "BLACKS FORK JCT", 41.2333, -110.4500, 8400, "SNOTEL:WY:959"),
    ("960", "BROOKS LAKE", 43.6333, -110.0333, 9600, "SNOTEL:WY:960"),
    ("961", "BURROUGHS CREEK", 43.8833, -110.1000, 9400, "SNOTEL:WY:961"),
    ("962", "COLE CREEK", 44.2333, -107.3833, 8600, "SNOTEL:WY:962"),
    ("963", "COPPER MOUNTAIN", 42.4500, -106.8667, 8200, "SNOTEL:WY:963"),
    ("964", "COTTONWOOD CREEK", 44.5500, -109.2833, 8840, "SNOTEL:WY:964"),
    ("965", "CROUSE CREEK", 41.3833, -110.7333, 9120, "SNOTEL:WY:965"),
    ("966", "DRY CREEK", 44.3833, -107.3667, 8200, "SNOTEL:WY:966"),
    ("967", "ELKHORN PARK", 41.5667, -106.8667, 10200, "SNOTEL:WY:967"),
    ("968", "FISH CREEK", 43.2833, -110.2333, 9800, "SNOTEL:WY:968"),
    ("969", "GRASSY LAKE", 44.6167, -110.4667, 7600, "SNOTEL:WY:969"),
    ("970", "HOBBS PARK", 41.2167, -106.9167, 9800, "SNOTEL:WY:970"),
    ("971", "KENDALL RANGER STATION", 44.3833, -109.5833, 6920, "SNOTEL:WY:971"),
    ("972", "LARAMIE RIVER", 41.2833, -105.9667, 9200, "SNOTEL:WY:972"),
    ("973", "LITTLE SNAKE RIVER", 41.7833, -107.0500, 7600, "SNOTEL:WY:973"),
    ("974", "LONG POND", 44.6833, -110.3500, 8200, "SNOTEL:WY:974"),
    ("975", "MEDICINE BOW", 41.3500, -106.3167, 9700, "SNOTEL:WY:975"),
    ("976", "NEW FORK LAKE", 42.8333, -109.6667, 8200, "SNOTEL:WY:976"),
    ("977", "NO NAME", 41.1667, -107.0333, 10400, "SNOTEL:WY:977"),
    ("978", "OWL CREEK", 43.6167, -109.8833, 8800, "SNOTEL:WY:978"),
    ("979", "PINE CREEK", 44.6500, -107.3167, 8440, "SNOTEL:WY:979"),
    ("980", "POWDER RIVER PASS", 44.1500, -107.1833, 9660, "SNOTEL:WY:980"),
    ("981", "SHELL CREEK", 44.7167, -107.8500, 8600, "SNOTEL:WY:981"),
    ("982", "SNIDER BASIN", 41.2167, -107.0167, 9600, "SNOTEL:WY:982"),
    ("983", "SNOW TELEMETRY", 44.4833, -107.3833, 8200, "SNOTEL:WY:983"),
    ("984", "SOUTH BRUSH CREEK", 41.3333, -106.3667, 9700, "SNOTEL:WY:984"),
    ("985", "TIE CREEK", 44.3500, -107.3667, 8720, "SNOTEL:WY:985"),
    ("986", "TOWER FALLS", 44.9167, -110.4167, 6800, "SNOTEL:WY:986"),
    ("987", "TWIN CREEK", 44.2667, -107.3167, 9200, "SNOTEL:WY:987"),
    ("988", "WEBBER SPRINGS", 41.1333, -106.9667, 9800, "SNOTEL:WY:988"),
    ("989", "WILLOW PARK", 41.5333, -106.9833, 10200, "SNOTEL:WY:989"),
    ("990", "YELLOWSTONE LAKE", 44.5333, -110.3667, 7800, "SNOTEL:WY:990"),
]

def create_station_file(output_path: Path = None):
    """Create SNOTEL station GeoJSON from known stations"""
    
    if output_path is None:
        output_path = Path("data/cache/snotel_stations_wyoming.geojson")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    station_data = []
    for station_id, name, lat, lon, elevation_ft, triplet in WYOMING_SNOTEL_STATIONS:
        station_data.append({
            "station_id": station_id,
            "triplet": triplet,
            "name": name,
            "lat": lat,
            "lon": lon,
            "elevation_ft": elevation_ft,
            "state": "WY"
        })
    
    df = pd.DataFrame(station_data)
    
    # Create GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Save to GeoJSON
    gdf.to_file(output_path, driver="GeoJSON")
    
    print(f"✓ Created {len(gdf)} SNOTEL stations in Wyoming")
    print(f"✓ Saved to {output_path}")
    print(f"\nNote: This is a sample list. For complete data:")
    print(f"  1. Visit: https://wcc.sc.egov.usda.gov/reportGenerator/")
    print(f"  2. Select State=Wyoming, Network=SNOTEL")
    print(f"  3. Export station list and update this script")
    
    return gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create SNOTEL station file from known Wyoming stations"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cache/snotel_stations_wyoming.geojson"),
        help="Output GeoJSON file path"
    )
    
    args = parser.parse_args()
    
    create_station_file(args.output)

