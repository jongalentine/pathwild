#!/usr/bin/env python3
"""
Map USDA SNOTEL station triplets to snotelr site IDs.

This script uses snotelr::snotel_info() to get all SNOTEL stations and maps them
to our USDA station triplets by matching station names and locations.

Usage:
    python scripts/map_snotel_station_ids.py [--stations-file PATH] [--output PATH]

Requirements:
    - R and snotelr package installed
    - rpy2 Python package
    - Station GeoJSON file from download_snotel_stations_manual.py
"""

import argparse
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from difflib import SequenceMatcher
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def normalize_name(name: str) -> str:
    """
    Normalize station name for better matching.
    Removes common suffixes and normalizes whitespace.
    """
    if pd.isna(name) or name is None:
        return ""
    
    name = str(name).strip().upper()
    
    # Remove common suffixes
    suffixes = [
        " SNOTEL", " SNOTEL SITE", " SNOTEL STATION",
        " RANGER STATION", " RS", " RAWS",
        " WY", " WYOMING"
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name)
    
    return name


def fuzzy_match_name(target_name: str, candidate_names: pd.Series, threshold: float = 0.8) -> pd.Series:
    """
    Find best fuzzy match for a station name.
    
    Args:
        target_name: Name to match
        candidate_names: Series of candidate names
        threshold: Minimum similarity threshold (0-1)
    
    Returns:
        Series of boolean values indicating matches
    """
    target_normalized = normalize_name(target_name)
    matches = pd.Series(False, index=candidate_names.index)
    
    for idx, candidate in candidate_names.items():
        if pd.isna(candidate):
            continue
        
        candidate_normalized = normalize_name(str(candidate))
        
        # Calculate similarity
        similarity = SequenceMatcher(None, target_normalized, candidate_normalized).ratio()
        
        # Also check if one name contains the other (for partial matches)
        if target_normalized in candidate_normalized or candidate_normalized in target_normalized:
            similarity = max(similarity, 0.85)
        
        matches.at[idx] = similarity >= threshold
    
    return matches


def map_snotel_station_ids(stations_file: Path, output_file: Path = None):
    """
    Map USDA station triplets to snotelr site IDs.
    
    Args:
        stations_file: Path to input station GeoJSON file
        output_file: Path to output GeoJSON file (default: overwrite input)
    """
    if output_file is None:
        output_file = stations_file
    
    print("=" * 70)
    print("SNOTEL Station ID Mapping")
    print("=" * 70)
    print()
    
    # Load station file
    print(f"Loading stations from: {stations_file}")
    if not stations_file.exists():
        print(f"Error: Station file not found: {stations_file}")
        sys.exit(1)
    
    stations_gdf = gpd.read_file(stations_file)
    print(f"  ✓ Loaded {len(stations_gdf)} stations")
    print()
    
    # Initialize R and snotelr
    print("Initializing R and snotelr...")
    try:
        import rpy2.robjects as ro  # type: ignore
        from rpy2.robjects.packages import importr  # type: ignore
        from rpy2.robjects.conversion import localconverter  # type: ignore
        from rpy2.robjects import pandas2ri  # type: ignore
        
        snotelr = importr('snotelr')
        print("  ✓ snotelr loaded successfully")
    except ImportError:
        print("  ❌ rpy2 not available. Install with: pip install rpy2")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ Error loading snotelr: {e}")
        print("  Install with: R -e \"install.packages('snotelr', repos='https://cloud.r-project.org')\"")
        sys.exit(1)
    
    print()
    
    # Get all SNOTEL stations from snotelr
    print("Fetching all SNOTEL stations from snotelr...")
    try:
        snotel_info = snotelr.snotel_info
        r_stations = snotel_info()
        
        # Convert R data frame to pandas
        with localconverter(pandas2ri.converter):
            stations_df = pandas2ri.rpy2py(r_stations)
        
        print(f"  ✓ Retrieved {len(stations_df)} stations from snotelr")
        print(f"  Columns: {list(stations_df.columns)}")
    except Exception as e:
        print(f"  ❌ Error fetching stations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    
    # Prepare snotelr stations for matching
    print("Preparing station data for matching...")
    # Filter to Wyoming stations for faster matching
    wy_stations_df = stations_df[stations_df['state'] == 'WY'].copy()
    print(f"  ✓ Found {len(wy_stations_df)} Wyoming stations in snotelr")
    
    # Normalize snotelr station names
    wy_stations_df['name_normalized'] = wy_stations_df['site_name'].apply(normalize_name)
    print()
    
    # Map stations
    print("Mapping stations using multiple strategies...")
    mapped_count = 0
    unmatched = []
    mapping_strategies = {}
    
    # Initialize mapping column (preserve existing mappings if any)
    if 'snotelr_site_id' not in stations_gdf.columns:
        stations_gdf['snotelr_site_id'] = None
    else:
        # Count existing mappings
        existing = stations_gdf['snotelr_site_id'].notna().sum()
        if existing > 0:
            print(f"  Preserving {existing} existing mappings")
    
    # Match by station name (case-insensitive, handle variations)
    for idx, station in stations_gdf.iterrows():
        # Skip if already mapped
        if pd.notna(stations_gdf.at[idx, 'snotelr_site_id']):
            continue
        
        station_name = str(station['name']).strip().upper()
        station_normalized = normalize_name(station_name)
        triplet = station['triplet']
        station_lat = station.get('lat', None)
        station_lon = station.get('lon', None)
        
        matches = pd.DataFrame()
        strategy = None
        
        # Strategy 1: Exact normalized name match
        if len(matches) == 0:
            name_matches = wy_stations_df[wy_stations_df['name_normalized'] == station_normalized]
            if len(name_matches) > 0:
                matches = name_matches
                strategy = "exact_normalized"
        
        # Strategy 2: Original exact match (case-insensitive)
        if len(matches) == 0:
            name_matches = wy_stations_df[
                wy_stations_df['site_name'].str.upper().str.strip() == station_name
            ]
            if len(name_matches) > 0:
                matches = name_matches
                strategy = "exact_original"
        
        # Strategy 3: Fuzzy string matching (high threshold)
        if len(matches) == 0:
            fuzzy_matches = fuzzy_match_name(station_name, wy_stations_df['site_name'], threshold=0.85)
            if fuzzy_matches.any():
                matches = wy_stations_df[fuzzy_matches]
                strategy = "fuzzy_high"
        
        # Strategy 4: Partial match (contains)
        if len(matches) == 0 and len(station_normalized.split()) >= 2:
            # Try matching with first 2-3 words
            name_parts = station_normalized.split()
            for n_words in range(min(3, len(name_parts)), 1, -1):
                partial_name = ' '.join(name_parts[:n_words])
                partial_matches = wy_stations_df[
                    wy_stations_df['name_normalized'].str.contains(partial_name, na=False, regex=False)
                ]
                if len(partial_matches) > 0:
                    matches = partial_matches
                    strategy = f"partial_{n_words}_words"
                    break
        
        # Strategy 5: Location-based matching (within 0.05 degrees ~5.5 km)
        if len(matches) == 0 and station_lat is not None and station_lon is not None:
            # Calculate distances
            wy_stations_dist = wy_stations_df.copy()
            wy_stations_dist['distance_deg'] = (
                ((wy_stations_dist['latitude'] - station_lat) ** 2) +
                ((wy_stations_dist['longitude'] - station_lon) ** 2)
            ) ** 0.5
            
            # Get closest within 0.05 degrees (~5.5 km)
            closest = wy_stations_dist[wy_stations_dist['distance_deg'] < 0.05].nsmallest(1, 'distance_deg')
            if len(closest) > 0:
                matches = closest
                strategy = "location_close"
        
        # Strategy 6: Location-based matching (relaxed, within 0.1 degrees ~11 km)
        if len(matches) == 0 and station_lat is not None and station_lon is not None:
            wy_stations_dist = wy_stations_df.copy()
            wy_stations_dist['distance_deg'] = (
                ((wy_stations_dist['latitude'] - station_lat) ** 2) +
                ((wy_stations_dist['longitude'] - station_lon) ** 2)
            ) ** 0.5
            
            closest = wy_stations_dist[wy_stations_dist['distance_deg'] < 0.1].nsmallest(1, 'distance_deg')
            if len(closest) > 0:
                matches = closest
                strategy = "location_relaxed"
        
        # Strategy 7: Fuzzy matching with lower threshold
        if len(matches) == 0:
            fuzzy_matches = fuzzy_match_name(station_name, wy_stations_df['site_name'], threshold=0.7)
            if fuzzy_matches.any():
                matches = wy_stations_df[fuzzy_matches]
                # Get best match by similarity
                if len(matches) > 1:
                    similarities = [
                        SequenceMatcher(None, normalize_name(station_name), normalize_name(name)).ratio()
                        for name in matches['site_name']
                    ]
                    best_idx = similarities.index(max(similarities))
                    matches = matches.iloc[[best_idx]]
                strategy = "fuzzy_low"
        
        if len(matches) > 0:
            # Use first/best match
            match_row = matches.iloc[0]
            snotelr_id = int(match_row['site_id'])
            snotelr_name = match_row['site_name']
            stations_gdf.at[idx, 'snotelr_site_id'] = snotelr_id
            mapped_count += 1
            mapping_strategies[station_name] = strategy
            print(f"  ✓ {station_name} → {snotelr_name} (site_id: {snotelr_id}, strategy: {strategy})")
        else:
            unmatched.append({
                'name': station_name,
                'triplet': triplet,
                'lat': station_lat,
                'lon': station_lon
            })
            print(f"  ⚠ {station_name}: No match found")
    
    print()
    print(f"Mapped {mapped_count} stations (including {existing if 'existing' in locals() else 0} existing)")
    
    # Print strategy summary
    if mapping_strategies:
        print("\nMapping strategies used:")
        strategy_counts = {}
        for strategy in mapping_strategies.values():
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        for strategy, count in sorted(strategy_counts.items()):
            print(f"  {strategy}: {count}")
    
    if unmatched:
        print(f"\n⚠ Unmatched stations ({len(unmatched)}):")
        for un in unmatched[:10]:
            print(f"  - {un['name']} ({un['triplet']}) at ({un['lat']}, {un['lon']})")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")
    
    # Show potential matches for unmatched stations
    if unmatched and len(wy_stations_df) > 0:
        print("\nSuggestions for unmatched stations:")
        for un in unmatched[:5]:
            station_lat = un.get('lat')
            station_lon = un.get('lon')
            if station_lat and station_lon:
                # Find nearby stations
                nearby = wy_stations_df.copy()
                nearby['distance_deg'] = (
                    ((nearby['latitude'] - station_lat) ** 2) +
                    ((nearby['longitude'] - station_lon) ** 2)
                ) ** 0.5
                nearby_sorted = nearby.nsmallest(3, 'distance_deg')
                if len(nearby_sorted) > 0:
                    print(f"\n  {un['name']} ({un['triplet']}):")
                    for _, nearby_station in nearby_sorted.iterrows():
                        dist_km = nearby_station['distance_deg'] * 111.0  # rough conversion
                        print(f"    - {nearby_station['site_name']} (site_id: {nearby_station['site_id']}, "
                              f"distance: ~{dist_km:.1f} km)")
    print()
    
    # Save updated file
    print(f"Saving mapped stations to: {output_file}")
    stations_gdf.to_file(output_file, driver="GeoJSON")
    print(f"  ✓ Saved {len(stations_gdf)} stations with snotelr site IDs")
    
    # Print summary
    mapped_stations = stations_gdf[stations_gdf['snotelr_site_id'].notna()]
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total stations: {len(stations_gdf)}")
    print(f"Mapped stations: {len(mapped_stations)}")
    print(f"Unmapped stations: {len(stations_gdf) - len(mapped_stations)}")
    print()
    print("Next steps:")
    print("  1. Review unmatched stations (may need manual mapping)")
    print("  2. Test with: python scripts/test_snotel_integration.py")
    print("  3. Verify data retrieval works with mapped IDs")
    
    return stations_gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map USDA SNOTEL station triplets to snotelr site IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--stations-file",
        type=Path,
        default=Path("data/cache/snotel_stations_wyoming.geojson"),
        help="Input station GeoJSON file"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GeoJSON file (default: overwrite input file)"
    )
    
    args = parser.parse_args()
    
    map_snotel_station_ids(args.stations_file, args.output)

