"""
Wyoming Hunt Area definitions and coordinates.

This module provides hunt area boundaries and center points for distance calculations.
"""

from typing import Dict, Tuple, Optional
from pathlib import Path
from shapely.geometry import Point, Polygon
import geopandas as gpd

# Hunt Area 048 (Upper Nowood) - Wyoming
# Exact polygon available from WGFD shapefile
# Shapefile location: data/raw/hunt_areas/Area_048.shp

# Center point calculated from actual shapefile bounds
AREA_048_CENTER: Tuple[float, float] = (43.4105, -107.5204)  # (lat, lon) - from shapefile
AREA_048_NAME: str = "Area 048"
AREA_048_FULL_NAME: str = "Wyoming Hunt Area 048 (Upper Nowood)"
AREA_048_HUNT_NAME: str = "Upper Nowood"
AREA_048_SIZE_SQ_MILES: float = 930.45  # From shapefile
AREA_048_SIZE_ACRES: float = 595488.0  # Calculated from square miles
AREA_048_HERD_UNIT: float = 322.0
AREA_048_HERD_NAME: str = "South Bighorn"
AREA_048_REGION: str = "Western"

# Exact bounding box from shapefile (WGS84)
AREA_048_BBOX = {
    "min_lat": 43.0707,
    "max_lat": 43.7503,
    "min_lon": -107.9076,
    "max_lon": -107.1333
}

def get_area_048_center() -> Tuple[float, float]:
    """Get the center coordinates of Area 048.
    
    Returns:
        Tuple of (latitude, longitude) in decimal degrees (WGS84)
    """
    return AREA_048_CENTER

def get_area_048_polygon() -> Optional[Polygon]:
    """
    Get the polygon boundary for Area 048.
    
    Loads the shapefile from data/raw/hunt_areas/Area_048.shp
    
    Returns:
        Shapely Polygon if shapefile is loaded, None otherwise
    """
    shapefile_path = Path(__file__).parent.parent.parent / "data" / "raw" / "hunt_areas" / "Area_048.shp"
    
    if not shapefile_path.exists():
        return None
    
    try:
        gdf = gpd.read_file(shapefile_path)
        if len(gdf) > 0:
            # Get the first (and only) geometry
            return gdf.geometry.iloc[0]
    except Exception as e:
        print(f"Error loading Area 048 polygon: {e}")
    
    return None

def load_area_048_shapefile(shapefile_path: Optional[str] = None) -> Optional[gpd.GeoDataFrame]:
    """
    Load Area 048 boundary from WGFD shapefile.
    
    Args:
        shapefile_path: Optional path to the Area 048 shapefile.
                       If None, uses default location: data/raw/hunt_areas/Area_048.shp
        
    Returns:
        GeoDataFrame with Area 048 boundary, or None if file not found
    """
    if shapefile_path is None:
        shapefile_path = str(Path(__file__).parent.parent.parent / "data" / "raw" / "hunt_areas" / "Area_048.shp")
    
    try:
        gdf = gpd.read_file(shapefile_path)
        # If loading from full ElkHuntAreas.shp, filter for Area 048
        if 'HUNTAREA' in gdf.columns and len(gdf) > 1:
            gdf = gdf[gdf['HUNTAREA'] == 48.0]
        return gdf
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return None

def point_in_area_048(lat: float, lon: float, polygon: Optional[Polygon] = None) -> bool:
    """
    Check if a point is within Area 048 boundaries.
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        polygon: Optional pre-loaded Area 048 polygon
        
    Returns:
        True if point is in Area 048, False otherwise
    """
    if polygon is None:
        polygon = get_area_048_polygon()
    
    if polygon is None:
        # Fall back to bounding box check
        bbox = AREA_048_BBOX
        return (bbox["min_lat"] <= lat <= bbox["max_lat"] and
                bbox["min_lon"] <= lon <= bbox["max_lon"])
    
    point = Point(lon, lat)  # Shapely uses (x, y) = (lon, lat)
    return polygon.contains(point)

# Hunt area registry (for future expansion)
HUNT_AREAS: Dict[str, Dict] = {
    "048": {
        "name": AREA_048_NAME,
        "full_name": AREA_048_FULL_NAME,
        "hunt_name": AREA_048_HUNT_NAME,
        "center": AREA_048_CENTER,
        "herd_unit": AREA_048_HERD_UNIT,
        "herd_name": AREA_048_HERD_NAME,
        "region": AREA_048_REGION,
        "size_sq_miles": AREA_048_SIZE_SQ_MILES,
        "size_acres": AREA_048_SIZE_ACRES,
        "bbox": AREA_048_BBOX,
        "shapefile": "data/raw/hunt_areas/Area_048.shp"
    }
}
