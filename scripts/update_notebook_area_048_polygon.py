#!/usr/bin/env python3
"""
Update notebook 02_explore_south_bighorn.ipynb to use Area 048 polygon
instead of just center point coordinates.
"""

import json
from pathlib import Path

def update_notebook():
    """Update the notebook to use Area 048 polygon"""
    
    notebook_path = Path("notebooks/02_explore_south_bighorn.ipynb")
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Update Cell 1 - Add import for hunt_areas module
    cell_1_source = [
        "import geopandas as gpd\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from pathlib import Path\n",
        "from shapely.geometry import Point\n",
        "from src.data.hunt_areas import load_area_048_shapefile, get_area_048_polygon\n",
        "\n",
        "# Set up paths\n",
        "DATA_DIR = Path(\"../data/raw\")\n",
        "BIGHORN_FILE = DATA_DIR / \"elk_southern_bighorn\" / \"Elk_WY_Bighorn_South_Routes_Ver1_2020.shp\"\n",
        "\n",
        "print(\"Loading South Bighorn migration routes...\")\n",
        "print(f\"File exists: {BIGHORN_FILE.exists()}\")\n",
        "\n",
        "# Load Area 048 polygon\n",
        "print(\"\\nLoading Area 048 hunt area boundary...\")\n",
        "area_048_gdf = load_area_048_shapefile()\n",
        "if area_048_gdf is not None:\n",
        "    # Convert to WGS84 if needed\n",
        "    if area_048_gdf.crs != 'EPSG:4326':\n",
        "        area_048_gdf = area_048_gdf.to_crs('EPSG:4326')\n",
        "    area_048_polygon = area_048_gdf.geometry.iloc[0]\n",
        "    area_048_center = area_048_gdf.geometry.centroid.iloc[0]\n",
        "    area_048_lat = area_048_center.y\n",
        "    area_048_lon = area_048_center.x\n",
        "    print(f\"  ✓ Area 048 loaded: {area_048_gdf['HUNTNAME'].iloc[0]}\")\n",
        "    print(f\"    Center: {area_048_lat:.4f}°N, {area_048_lon:.4f}°W\")\n",
        "    print(f\"    Size: {area_048_gdf['SqMiles'].iloc[0]:.2f} sq miles\")\n",
        "else:\n",
        "    print(\"  ⚠️  Could not load Area 048 shapefile\")\n",
        "    area_048_polygon = None\n",
        "    area_048_lat, area_048_lon = 43.4105, -107.5204  # Fallback center\n"
    ]
    
    # Update Cell 9 - Distance calculation using polygon
    cell_9_source = [
        "# Convert to WGS84 (lat/lon) if needed\n",
        "if gdf.crs != 'EPSG:4326':\n",
        "    print(f\"Converting from {gdf.crs} to EPSG:4326 (WGS84)...\")\n",
        "    gdf_wgs84 = gdf.to_crs('EPSG:4326')\n",
        "    points_gdf_wgs84 = points_gdf.to_crs('EPSG:4326')\n",
        "else:\n",
        "    gdf_wgs84 = gdf\n",
        "    points_gdf_wgs84 = points_gdf\n",
        "\n",
        "# Extract coordinates\n",
        "points_gdf_wgs84['latitude'] = points_gdf_wgs84.geometry.y\n",
        "points_gdf_wgs84['longitude'] = points_gdf_wgs84.geometry.x\n",
        "\n",
        "print(f\"\\nSpatial Coverage:\")\n",
        "print(f\"  Latitude range: {points_gdf_wgs84['latitude'].min():.4f} to {points_gdf_wgs84['latitude'].max():.4f}\")\n",
        "print(f\"  Longitude range: {points_gdf_wgs84['longitude'].min():.4f} to {points_gdf_wgs84['longitude'].max():.4f}\")\n",
        "\n",
        "# Calculate distances to Area 048 polygon boundary\n",
        "if area_048_polygon is not None:\n",
        "    print(f\"\\nCalculating distances to Area 048 boundary...\")\n",
        "    \n",
        "    def distance_to_polygon(point_geom, polygon):\n",
        "        \"\"\"Calculate distance from point to polygon boundary (km)\"\"\"\n",
        "        # Distance in degrees (approximate)\n",
        "        dist_deg = point_geom.distance(polygon.boundary)\n",
        "        # Convert to km (rough approximation: 1 degree ≈ 111 km)\n",
        "        # More accurate: use UTM for distance calculation\n",
        "        from pyproj import Transformer\n",
        "        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32612', always_xy=True)\n",
        "        point_utm = transformer.transform(point_geom.x, point_geom.y)\n",
        "        polygon_utm = gpd.GeoSeries([polygon], crs='EPSG:4326').to_crs('EPSG:32612')\n",
        "        point_shapely = Point(point_utm[0], point_utm[1])\n",
        "        dist_m = point_shapely.distance(polygon_utm.geometry.iloc[0].boundary)\n",
        "        return dist_m / 1000  # Convert meters to km\n",
        "    \n",
        "    # Calculate distance to polygon boundary for each point\n",
        "    points_gdf_wgs84['distance_to_area_048_km'] = points_gdf_wgs84.geometry.apply(\n",
        "        lambda geom: distance_to_polygon(geom, area_048_polygon)\n",
        "    )\n",
        "    \n",
        "    # Check which points are inside Area 048\n",
        "    points_gdf_wgs84['inside_area_048'] = points_gdf_wgs84.geometry.apply(\n",
        "        lambda geom: area_048_polygon.contains(geom)\n",
        "    )\n",
        "    \n",
        "    # For points inside, set distance to 0\n",
        "    points_gdf_wgs84.loc[points_gdf_wgs84['inside_area_048'], 'distance_to_area_048_km'] = 0.0\n",
        "    \n",
        "    print(f\"\\nProximity to Area 048:\")\n",
        "    print(f\"  Points inside Area 048: {points_gdf_wgs84['inside_area_048'].sum()} ({(points_gdf_wgs84['inside_area_048'].sum() / len(points_gdf_wgs84) * 100):.1f}%)\")\n",
        "    print(f\"  Minimum distance (outside): {points_gdf_wgs84[~points_gdf_wgs84['inside_area_048']]['distance_to_area_048_km'].min():.2f} km\")\n",
        "    print(f\"  Maximum distance: {points_gdf_wgs84['distance_to_area_048_km'].max():.2f} km\")\n",
        "    print(f\"  Average distance: {points_gdf_wgs84['distance_to_area_048_km'].mean():.2f} km\")\n",
        "    print(f\"  Points within 50km: {(points_gdf_wgs84['distance_to_area_048_km'] <= 50).sum()} ({(points_gdf_wgs84['distance_to_area_048_km'] <= 50).sum() / len(points_gdf_wgs84) * 100):.1f}%)\")\n",
        "else:\n",
        "    # Fallback to center point calculation\n",
        "    from math import radians, sin, cos, sqrt, atan2\n",
        "    \n",
        "    def haversine_distance(lat1, lon1, lat2, lon2):\n",
        "        R = 6371  # Earth radius in km\n",
        "        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])\n",
        "        dlat = lat2 - lat1\n",
        "        dlon = lon2 - lon1\n",
        "        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2\n",
        "        c = 2 * atan2(sqrt(a), sqrt(1-a))\n",
        "        return R * c\n",
        "    \n",
        "    points_gdf_wgs84['distance_to_area_048_km'] = points_gdf_wgs84.apply(\n",
        "        lambda row: haversine_distance(row['latitude'], row['longitude'], area_048_lat, area_048_lon),\n",
        "        axis=1\n",
        "    )\n",
        "    points_gdf_wgs84['inside_area_048'] = False\n",
        "    print(f\"\\n⚠️  Using center point fallback (polygon not loaded)\")\n"
    ]
    
    # Update Cell 11 - Visualization with polygon
    cell_11_source = [
        "# Plot the migration routes\n",
        "fig, ax = plt.subplots(figsize=(14, 12))\n",
        "\n",
        "# Plot routes\n",
        "gdf_wgs84.plot(ax=ax, linewidth=1.5, alpha=0.7, color='blue', label='Migration Routes')\n",
        "\n",
        "# Plot Area 048 polygon boundary\n",
        "if area_048_polygon is not None:\n",
        "    area_048_gdf.plot(ax=ax, color='none', edgecolor='red', linewidth=3, linestyle='-', label='Area 048 Boundary', alpha=0.8)\n",
        "    # Fill with slight transparency\n",
        "    area_048_gdf.plot(ax=ax, color='red', alpha=0.1, edgecolor='none')\n",
        "    \n",
        "    # Mark center point\n",
        "    area_048_center_point = gpd.GeoDataFrame(\n",
        "        [{'name': 'Area 048 Center'}],\n",
        "        geometry=[Point(area_048_lon, area_048_lat)],\n",
        "        crs='EPSG:4326'\n",
        "    )\n",
        "    area_048_center_point.plot(ax=ax, color='darkred', markersize=150, marker='*', label='Area 048 Center')\n",
        "else:\n",
        "    # Fallback: just show center point\n",
        "    area_048_point = gpd.GeoDataFrame(\n",
        "        [{'name': 'Area 048'}],\n",
        "        geometry=[Point(area_048_lon, area_048_lat)],\n",
        "        crs='EPSG:4326'\n",
        "    )\n",
        "    area_048_point.plot(ax=ax, color='red', markersize=200, marker='*', label='Area 048')\n",
        "\n",
        "# Add a circle showing 50km radius around Area 048 center (for reference)\n",
        "from shapely.ops import transform\n",
        "import pyproj\n",
        "\n",
        "def create_circle(lat, lon, radius_km):\n",
        "    \"\"\"Create a circle with radius in km\"\"\"\n",
        "    # Use UTM for accurate distance (Wyoming is in UTM Zone 12N)\n",
        "    utm = pyproj.Proj(proj='utm', zone=12, ellps='WGS84')\n",
        "    wgs84 = pyproj.Proj(proj='latlong', ellps='WGS84')\n",
        "    \n",
        "    # Convert center to UTM\n",
        "    center_x, center_y = pyproj.transform(wgs84, utm, lon, lat)\n",
        "    \n",
        "    # Create circle in UTM (radius in meters)\n",
        "    circle = Point(center_x, center_y).buffer(radius_km * 1000)\n",
        "    \n",
        "    # Convert back to WGS84\n",
        "    circle_wgs84 = transform(\n",
        "        pyproj.Transformer.from_proj(utm, wgs84, always_xy=True).transform,\n",
        "        circle\n",
        "    )\n",
        "    \n",
        "    return circle_wgs84\n",
        "\n",
        "circle_50km = create_circle(area_048_lat, area_048_lon, 50)\n",
        "circle_gdf = gpd.GeoDataFrame([{'radius': '50km'}], geometry=[circle_50km], crs='EPSG:4326')\n",
        "circle_gdf.plot(ax=ax, color='none', edgecolor='orange', linewidth=2, linestyle='--', label='50km radius', alpha=0.5)\n",
        "\n",
        "ax.set_title('South Bighorn Elk Migration Routes - Area 048 Boundary', fontsize=16, fontweight='bold')\n",
        "ax.set_xlabel('Longitude', fontsize=12)\n",
        "ax.set_ylabel('Latitude', fontsize=12)\n",
        "ax.legend(fontsize=10)\n",
        "ax.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.savefig('../data/processed/south_bighorn_routes_area_048.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n✓ Map saved to data/processed/south_bighorn_routes_area_048.png\")\n",
        "if area_048_polygon is not None:\n",
        "    print(\"  Red polygon = Area 048 boundary\")\n",
        "    print(\"  Points inside polygon are within Area 048\")\n"
    ]
    
    # Update Cell 15 - Add inside_area_048 column
    cell_15_source = [
        "# Create a simplified dataframe for PathWild\n",
        "# This will be used to create training examples\n",
        "\n",
        "pathwild_data = pd.DataFrame({\n",
        "    'latitude': points_gdf_wgs84['latitude'],\n",
        "    'longitude': points_gdf_wgs84['longitude'],\n",
        "    'route_id': points_gdf_wgs84['route_id'],\n",
        "    'distance_to_area_048_km': points_gdf_wgs84['distance_to_area_048_km']\n",
        "})\n",
        "\n",
        "# Add inside_area_048 flag if available\n",
        "if 'inside_area_048' in points_gdf_wgs84.columns:\n",
        "    pathwild_data['inside_area_048'] = points_gdf_wgs84['inside_area_048']\n",
        "\n",
        "# Add any other relevant columns from the original data\n",
        "for col in gdf.columns:\n",
        "    if col != 'geometry' and col not in pathwild_data.columns:\n",
        "        # Map route-level attributes to points\n",
        "        route_values = gdf[col].to_dict()\n",
        "        pathwild_data[col] = pathwild_data['route_id'].map(route_values)\n",
        "\n",
        "print(\"PathWild-ready dataset:\")\n",
        "print(f\"  Shape: {pathwild_data.shape}\")\n",
        "print(f\"  Columns: {list(pathwild_data.columns)}\")\n",
        "print(f\"\\nFirst few rows:\")\n",
        "print(pathwild_data.head())\n",
        "\n",
        "if 'inside_area_048' in pathwild_data.columns:\n",
        "    print(f\"\\nPoints inside Area 048: {pathwild_data['inside_area_048'].sum()} ({(pathwild_data['inside_area_048'].sum() / len(pathwild_data) * 100):.1f}%)\")\n",
        "\n",
        "# Save to CSV for later use\n",
        "output_file = Path(\"../data/processed/south_bighorn_points.csv\")\n",
        "output_file.parent.mkdir(parents=True, exist_ok=True)\n",
        "pathwild_data.to_csv(output_file, index=False)\n",
        "print(f\"\\n✓ Saved to {output_file}\")\n"
    ]
    
    # Update Cell 17 - Summary with polygon info
    cell_17_source = [
        "print(\"=\" * 60)\n",
        "print(\"SOUTH BIGHORN DATASET SUMMARY\")\n",
        "print(\"=\" * 60)\n",
        "print(f\"\\nTotal migration routes: {len(gdf)}\")\n",
        "print(f\"Total points extracted: {len(points_gdf_wgs84):,}\")\n",
        "print(f\"\\nGeographic coverage:\")\n",
        "print(f\"  Latitude: {points_gdf_wgs84['latitude'].min():.4f}° to {points_gdf_wgs84['latitude'].max():.4f}°\")\n",
        "print(f\"  Longitude: {points_gdf_wgs84['longitude'].min():.4f}° to {points_gdf_wgs84['longitude'].max():.4f}°\")\n",
        "print(f\"\\nProximity to Area 048:\")\n",
        "if 'inside_area_048' in points_gdf_wgs84.columns:\n",
        "    inside_count = points_gdf_wgs84['inside_area_048'].sum()\n",
        "    print(f\"  Points inside Area 048: {inside_count:,} ({inside_count / len(points_gdf_wgs84) * 100:.1f}%)\")\n",
        "    print(f\"  Points within 50km: {(points_gdf_wgs84['distance_to_area_048_km'] <= 50).sum():,} ({(points_gdf_wgs84['distance_to_area_048_km'] <= 50).sum() / len(points_gdf_wgs84) * 100):.1f}%)\")\n",
        "    if inside_count > 0:\n",
        "        print(f\"  Closest point (outside): {points_gdf_wgs84[~points_gdf_wgs84['inside_area_048']]['distance_to_area_048_km'].min():.2f} km away\")\n",
        "    else:\n",
        "        print(f\"  Closest point: {points_gdf_wgs84['distance_to_area_048_km'].min():.2f} km away\")\n",
        "else:\n",
        "    print(f\"  Points within 50km: {(points_gdf_wgs84['distance_to_area_048_km'] <= 50).sum():,}\")\n",
        "    print(f\"  Closest point: {points_gdf_wgs84['distance_to_area_048_km'].min():.2f} km away\")\n",
        "\n",
        "if area_048_polygon is not None:\n",
        "    print(f\"\\nArea 048 Information:\")\n",
        "    print(f\"  Hunt Name: {area_048_gdf['HUNTNAME'].iloc[0]}\")\n",
        "    print(f\"  Size: {area_048_gdf['SqMiles'].iloc[0]:.2f} sq miles\")\n",
        "    print(f\"  Herd: {area_048_gdf['HERDNAME'].iloc[0]}\")\n",
        "\n",
        "print(f\"\\nNext steps:\")\n",
        "print(\"  1. Review the route attributes to understand seasonal patterns\")\n",
        "print(\"  2. Integrate with DataContextBuilder to add environmental features\")\n",
        "print(\"  3. Create training dataset with positive examples (route points)\")\n",
        "print(\"  4. Generate negative examples (random points not on routes)\")\n",
        "print(\"  5. Train XGBoost model with your heuristics + ML features\")\n"
    ]
    
    # Update the cells
    notebook['cells'][1]['source'] = cell_1_source
    notebook['cells'][9]['source'] = cell_9_source
    notebook['cells'][11]['source'] = cell_11_source
    notebook['cells'][15]['source'] = cell_15_source
    notebook['cells'][17]['source'] = cell_17_source
    
    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Updated {notebook_path}")
    print("  - Added Area 048 polygon loading")
    print("  - Updated distance calculation to use polygon boundary")
    print("  - Added inside_area_048 flag")
    print("  - Updated visualization to show polygon boundary")
    print("  - Updated summary with polygon information")

if __name__ == "__main__":
    update_notebook()
