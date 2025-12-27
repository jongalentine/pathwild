#!/usr/bin/env python3
"""
Update the Southern GYE notebook to match the actual downloaded data structure.
The data uses UTM coordinates (Easting/Northing) instead of lat/lon.
"""

import json
from pathlib import Path

# Read existing notebook
notebook_path = Path("notebooks/04_explore_southern_gye.ipynb")
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Update cells to handle UTM coordinates
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Exploring Southern GYE Elk GPS Collar Data (2007-2015)\n",
            "\n",
            "This notebook explores the Southern Greater Yellowstone Ecosystem GPS collar dataset - excellent for large sample size training!\n",
            "\n",
            "**Dataset Info:**\n",
            "- **Location:** 22 Wyoming winter supplemental feedgrounds\n",
            "- **Coverage:** 288 adult and yearling female elk, 2007-2015\n",
            "- **Data:** GPS locations during brucellosis risk period (February-July)\n",
            "- **Use Case:** Large sample size, diverse conditions, statistical robustness\n",
            "- **Note:** ~200 miles from Area 048, but provides excellent training data\n",
            "\n",
            "**Data Format:** UTM coordinates (Easting/Northing) in Zone 12N - will be converted to lat/lon"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import geopandas as gpd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from pathlib import Path\n",
            "from shapely.geometry import Point\n",
            "import pyproj\n",
            "\n",
            "# Set up paths\n",
            "DATA_DIR = Path(\"../data/raw\")\n",
            "GYE_DIR = DATA_DIR / \"elk_southern_gye\"\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"SOUTHERN GYE DATASET\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"\\nData directory: {GYE_DIR}\")\n",
            "print(f\"Directory exists: {GYE_DIR.exists()}\")\n",
            "\n",
            "# Look for data files\n",
            "if GYE_DIR.exists():\n",
            "    files = list(GYE_DIR.glob(\"*.csv\"))\n",
            "    print(f\"\\nCSV files found: {len(files)}\")\n",
            "    for f in files:\n",
            "        print(f\"  - {f.name}\")\n",
            "else:\n",
            "    print(\"\\n⚠️  Directory doesn't exist yet!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 1: Load the Data\n",
            "\n",
            "The dataset uses **UTM coordinates** (Easting/Northing) in Zone 12N. We'll convert these to lat/lon."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Find and load the CSV file\n",
            "csv_files = list(GYE_DIR.glob(\"*.csv\"))\n",
            "\n",
            "if csv_files:\n",
            "    csv_file = csv_files[0]\n",
            "    print(f\"Loading: {csv_file.name}\")\n",
            "    df = pd.read_csv(csv_file)\n",
            "    \n",
            "    print(f\"\\n✓ Data loaded successfully!\")\n",
            "    print(f\"  Shape: {df.shape}\")\n",
            "    print(f\"  Columns: {list(df.columns)}\")\n",
            "    print(f\"\\nFirst few rows:\")\n",
            "    print(df.head())\n",
            "    \n",
            "    # Expected columns: AID, Easting, Northing, Date_Time_MST, Feedground\n",
            "    print(f\"\\nColumn check:\")\n",
            "    expected_cols = ['AID', 'Easting', 'Northing', 'Date_Time_MST', 'Feedground']\n",
            "    for col in expected_cols:\n",
            "        if col in df.columns:\n",
            "            print(f\"  ✓ {col}\")\n",
            "        else:\n",
            "            print(f\"  ✗ {col} (missing)\")\n",
            "else:\n",
            "    print(\"⚠️  No CSV files found!\")\n",
            "    df = None"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 2: Convert UTM Coordinates to Lat/Lon\n",
            "\n",
            "The data uses UTM Zone 12N coordinates. We'll convert to WGS84 (lat/lon) for analysis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if df is not None:\n",
            "    print(\"=\" * 60)\n",
            "    print(\"CONVERTING UTM TO LAT/LON\")\n",
            "    print(\"=\" * 60)\n",
            "    \n",
            "    # UTM Zone 12N (Wyoming)\n",
            "    utm_zone = 12\n",
            "    utm_crs = f'EPSG:326{utm_zone}'  # UTM Zone 12N\n",
            "    wgs84_crs = 'EPSG:4326'  # WGS84 lat/lon\n",
            "    \n",
            "    print(f\"\\nUTM Zone: {utm_zone}N\")\n",
            "    print(f\"Converting {len(df):,} points...\")\n",
            "    \n",
            "    # Create GeoDataFrame from UTM coordinates\n",
            "    gdf_utm = gpd.GeoDataFrame(\n",
            "        df,\n",
            "        geometry=gpd.points_from_xy(df['Easting'], df['Northing']),\n",
            "        crs=utm_crs\n",
            "    )\n",
            "    \n",
            "    # Convert to WGS84 (lat/lon)\n",
            "    gdf_wgs84 = gdf_utm.to_crs(wgs84_crs)\n",
            "    \n",
            "    # Extract lat/lon\n",
            "    gdf_wgs84['latitude'] = gdf_wgs84.geometry.y\n",
            "    gdf_wgs84['longitude'] = gdf_wgs84.geometry.x\n",
            "    \n",
            "    print(f\"\\n✓ Conversion complete!\")\n",
            "    print(f\"\\nSample converted coordinates:\")\n",
            "    print(f\"  UTM: Easting={df['Easting'].iloc[0]:.2f}, Northing={df['Northing'].iloc[0]:.2f}\")\n",
            "    print(f\"  WGS84: Lat={gdf_wgs84['latitude'].iloc[0]:.4f}°, Lon={gdf_wgs84['longitude'].iloc[0]:.4f}°\")\n",
            "    \n",
            "    # Verify coordinates are in Wyoming range\n",
            "    lat_range = (gdf_wgs84['latitude'].min(), gdf_wgs84['latitude'].max())\n",
            "    lon_range = (gdf_wgs84['longitude'].min(), gdf_wgs84['longitude'].max())\n",
            "    print(f\"\\nCoordinate ranges:\")\n",
            "    print(f\"  Latitude: {lat_range[0]:.4f}° to {lat_range[1]:.4f}°\")\n",
            "    print(f\"  Longitude: {lon_range[0]:.4f}° to {lon_range[1]:.4f}°\")\n",
            "    \n",
            "    if 41 <= lat_range[0] <= 45 and -111 <= lon_range[0] <= -104:\n",
            "        print(f\"  ✓ Coordinates are in Wyoming range\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Inspect Dataset Structure"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if 'gdf_wgs84' in locals() and gdf_wgs84 is not None:\n",
            "    print(\"=\" * 60)\n",
            "    print(\"DATASET STRUCTURE\")\n",
            "    print(\"=\" * 60)\n",
            "    print(f\"\\nShape: {gdf_wgs84.shape}\")\n",
            "    print(f\"Columns: {list(gdf_wgs84.columns)}\")\n",
            "    print(f\"\\nData types:\")\n",
            "    print(gdf_wgs84.dtypes)\n",
            "    print(f\"\\nMissing values:\")\n",
            "    missing = gdf_wgs84.isnull().sum()\n",
            "    if missing.sum() > 0:\n",
            "        for col, count in missing[missing > 0].items():\n",
            "            print(f\"  {col}: {count} ({count/len(gdf_wgs84)*100:.1f}%)\")\n",
            "    else:\n",
            "        print(\"  ✓ No missing values!\")\n",
            "    \n",
            "    print(f\"\\nUnique values:\")\n",
            "    print(f\"  Unique elk (AID): {gdf_wgs84['AID'].nunique()}\")\n",
            "    print(f\"  Unique feedgrounds: {gdf_wgs84['Feedground'].nunique()}\")\n",
            "    print(f\"  Feedgrounds: {sorted(gdf_wgs84['Feedground'].unique())}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Analyze Spatial Coverage"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if 'gdf_wgs84' in locals() and gdf_wgs84 is not None:\n",
            "    print(\"=\" * 60)\n",
            "    print(\"SPATIAL COVERAGE\")\n",
            "    print(\"=\" * 60)\n",
            "    print(f\"\\nLatitude: {gdf_wgs84['latitude'].min():.4f}° to {gdf_wgs84['latitude'].max():.4f}°\")\n",
            "    print(f\"Longitude: {gdf_wgs84['longitude'].min():.4f}° to {gdf_wgs84['longitude'].max():.4f}°\")\n",
            "    \n",
            "    # Distance to Area 048\n",
            "    area_048_lat, area_048_lon = 41.835, -106.425\n",
            "    \n",
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
            "    gdf_wgs84['distance_to_area_048_km'] = gdf_wgs84.apply(\n",
            "        lambda row: haversine_distance(row['latitude'], row['longitude'], area_048_lat, area_048_lon),\n",
            "        axis=1\n",
            "    )\n",
            "    \n",
            "    print(f\"\\nProximity to Area 048:\")\n",
            "    print(f\"  Min distance: {gdf_wgs84['distance_to_area_048_km'].min():.2f} km\")\n",
            "    print(f\"  Max distance: {gdf_wgs84['distance_to_area_048_km'].max():.2f} km\")\n",
            "    print(f\"  Avg distance: {gdf_wgs84['distance_to_area_048_km'].mean():.2f} km\")\n",
            "    print(f\"  Points within 200km: {(gdf_wgs84['distance_to_area_048_km'] <= 200).sum():,} ({(gdf_wgs84['distance_to_area_048_km'] <= 200).sum() / len(gdf_wgs84) * 100:.1f}%)\")\n",
            "    print(f\"\\n⚠️  Note: Southern GYE is ~200 miles from Area 048.\")\n",
            "    print(f\"   This data is valuable for large sample size training.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 5: Analyze Temporal Patterns"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if 'gdf_wgs84' in locals() and gdf_wgs84 is not None:\n",
            "    print(\"=\" * 60)\n",
            "    print(\"TEMPORAL ANALYSIS\")\n",
            "    print(\"=\" * 60)\n",
            "    \n",
            "    # Parse date column (format: M/D/YYYY H:MM)\n",
            "    try:\n",
            "        gdf_wgs84['date'] = pd.to_datetime(gdf_wgs84['Date_Time_MST'], format='%m/%d/%Y %H:%M')\n",
            "        gdf_wgs84['year'] = gdf_wgs84['date'].dt.year\n",
            "        gdf_wgs84['month'] = gdf_wgs84['date'].dt.month\n",
            "        gdf_wgs84['day_of_year'] = gdf_wgs84['date'].dt.dayofyear\n",
            "        \n",
            "        print(f\"\\nDate range: {gdf_wgs84['date'].min()} to {gdf_wgs84['date'].max()}\")\n",
            "        print(f\"Total days: {(gdf_wgs84['date'].max() - gdf_wgs84['date'].min()).days}\")\n",
            "        \n",
            "        print(f\"\\nYear distribution:\")\n",
            "        for year, count in gdf_wgs84['year'].value_counts().sort_index().items():\n",
            "            print(f\"  {int(year)}: {count:,} points ({count/len(gdf_wgs84)*100:.1f}%)\")\n",
            "        \n",
            "        print(f\"\\nMonth distribution:\")\n",
            "        for month, count in gdf_wgs84['month'].value_counts().sort_index().items():\n",
            "            month_name = pd.to_datetime(f\"2020-{month}-01\").strftime(\"%B\")\n",
            "            print(f\"  {month_name} ({month}): {count:,} points ({count/len(gdf_wgs84)*100:.1f}%)\")\n",
            "        \n",
            "        # Note: Data is Feb-July (brucellosis risk period)\n",
            "        print(f\"\\n📋 Note: This dataset focuses on Feb-July (brucellosis risk period)\")\n",
            "        print(f\"   October data may be limited, but still valuable for general patterns.\")\n",
            "        \n",
            "        # Check October data\n",
            "        october_points = gdf_wgs84[gdf_wgs84['month'] == 10]\n",
            "        if len(october_points) > 0:\n",
            "            print(f\"\\n🎯 October data: {len(october_points):,} points ({len(october_points)/len(gdf_wgs84)*100:.1f}%)\")\n",
            "        else:\n",
            "            print(f\"\\n⚠️  No October data (expected - dataset focuses on Feb-July)\")\n",
            "            \n",
            "    except Exception as e:\n",
            "        print(f\"⚠️  Could not parse dates: {e}\")\n",
            "        print(f\"   Date format: {gdf_wgs84['Date_Time_MST'].iloc[0]}\")\n",
            "        print(f\"   Try adjusting the date format string if needed.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 6: Analyze Elk Individual Patterns"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if 'gdf_wgs84' in locals() and gdf_wgs84 is not None:\n",
            "    print(\"=\" * 60)\n",
            "    print(\"ELK INDIVIDUAL ANALYSIS\")\n",
            "    print(\"=\" * 60)\n",
            "    \n",
            "    print(f\"\\nTotal unique elk (AID): {gdf_wgs84['AID'].nunique()}\")\n",
            "    print(f\"Total GPS points: {len(gdf_wgs84):,}\")\n",
            "    print(f\"Average points per elk: {len(gdf_wgs84) / gdf_wgs84['AID'].nunique():.0f}\")\n",
            "    \n",
            "    points_per_elk = gdf_wgs84['AID'].value_counts()\n",
            "    print(f\"\\nPoints per elk:\")\n",
            "    print(f\"  Minimum: {points_per_elk.min():,}\")\n",
            "    print(f\"  Maximum: {points_per_elk.max():,}\")\n",
            "    print(f\"  Mean: {points_per_elk.mean():.0f}\")\n",
            "    print(f\"  Median: {points_per_elk.median():.0f}\")\n",
            "    \n",
            "    print(f\"\\nTop 5 elk by point count:\")\n",
            "    for elk_id, count in points_per_elk.head().items():\n",
            "        print(f\"  Elk {elk_id}: {count:,} points\")\n",
            "    \n",
            "    # Feedground analysis\n",
            "    print(f\"\\nFeedground distribution:\")\n",
            "    feedground_counts = gdf_wgs84['Feedground'].value_counts()\n",
            "    for feedground, count in feedground_counts.head(10).items():\n",
            "        print(f\"  {feedground}: {count:,} points ({count/len(gdf_wgs84)*100:.1f}%)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 7: Prepare Data for PathWild Integration"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if 'gdf_wgs84' in locals() and gdf_wgs84 is not None:\n",
            "    # Create PathWild-ready dataset\n",
            "    pathwild_data = pd.DataFrame({\n",
            "        'latitude': gdf_wgs84['latitude'],\n",
            "        'longitude': gdf_wgs84['longitude'],\n",
            "        'distance_to_area_048_km': gdf_wgs84['distance_to_area_048_km'],\n",
            "        'elk_id': gdf_wgs84['AID'],\n",
            "        'feedground': gdf_wgs84['Feedground']\n",
            "    })\n",
            "    \n",
            "    # Add temporal info if available\n",
            "    if 'date' in gdf_wgs84.columns:\n",
            "        pathwild_data['date'] = gdf_wgs84['date']\n",
            "        pathwild_data['year'] = gdf_wgs84['year']\n",
            "        pathwild_data['month'] = gdf_wgs84['month']\n",
            "        pathwild_data['day_of_year'] = gdf_wgs84['day_of_year']\n",
            "    \n",
            "    # Add original UTM coordinates (useful for reference)\n",
            "    pathwild_data['utm_easting'] = gdf_wgs84['Easting']\n",
            "    pathwild_data['utm_northing'] = gdf_wgs84['Northing']\n",
            "    \n",
            "    print(\"=\" * 60)\n",
            "    print(\"PATHWILD-READY DATASET\")\n",
            "    print(\"=\" * 60)\n",
            "    print(f\"\\nShape: {pathwild_data.shape}\")\n",
            "    print(f\"Columns: {list(pathwild_data.columns)}\")\n",
            "    print(f\"\\nFirst few rows:\")\n",
            "    print(pathwild_data.head())\n",
            "    \n",
            "    # Save to CSV\n",
            "    output_file = Path(\"../data/processed/southern_gye_points.csv\")\n",
            "    output_file.parent.mkdir(parents=True, exist_ok=True)\n",
            "    pathwild_data.to_csv(output_file, index=False)\n",
            "    print(f\"\\n✓ Saved to {output_file}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 8: Summary and Next Steps"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if 'gdf_wgs84' in locals() and gdf_wgs84 is not None:\n",
            "    print(\"=\" * 60)\n",
            "    print(\"SOUTHERN GYE DATASET SUMMARY\")\n",
            "    print(\"=\" * 60)\n",
            "    print(f\"\\nTotal GPS points: {len(gdf_wgs84):,}\")\n",
            "    print(f\"Unique elk: {gdf_wgs84['AID'].nunique()}\")\n",
            "    print(f\"Unique feedgrounds: {gdf_wgs84['Feedground'].nunique()}\")\n",
            "    print(f\"\\nGeographic coverage:\")\n",
            "    print(f\"  Latitude: {gdf_wgs84['latitude'].min():.4f}° to {gdf_wgs84['latitude'].max():.4f}°\")\n",
            "    print(f\"  Longitude: {gdf_wgs84['longitude'].min():.4f}° to {gdf_wgs84['longitude'].max():.4f}°\")\n",
            "    print(f\"\\nProximity to Area 048:\")\n",
            "    print(f\"  Average distance: {gdf_wgs84['distance_to_area_048_km'].mean():.2f} km\")\n",
            "    \n",
            "    if 'year' in gdf_wgs84.columns:\n",
            "        print(f\"\\nTemporal coverage:\")\n",
            "        print(f\"  Years: {sorted(gdf_wgs84['year'].unique())}\")\n",
            "        print(f\"  Months: {sorted(gdf_wgs84['month'].unique())}\")\n",
            "    \n",
            "    print(f\"\\n📋 Key Insights:\")\n",
            "    print(f\"  ✓ LARGE sample size ({len(gdf_wgs84):,} points from {gdf_wgs84['AID'].nunique()} elk)\")\n",
            "    print(f\"  ✓ Excellent for statistical robustness\")\n",
            "    print(f\"  ✓ Diverse conditions across {gdf_wgs84['Feedground'].nunique()} feedgrounds\")\n",
            "    print(f\"  ⚠️  Geographic distance from Area 048 (~200 miles)\")\n",
            "    print(f\"  ⚠️  Data focuses on Feb-July (brucellosis period)\")\n",
            "    print(f\"  → Best used for large-scale training and generalization\")\n",
            "    \n",
            "    print(f\"\\nNext steps:\")\n",
            "    print(\"  1. Combine with South Bighorn + National Elk Refuge data\")\n",
            "    print(\"  2. Use for large sample size training\")\n",
            "    print(\"  3. Integrate with DataContextBuilder to add environmental features\")\n",
            "    print(\"  4. Create training dataset with positive examples (GPS points)\")\n",
            "    print(\"  5. Generate negative examples (random points)\")\n",
            "    print(\"  6. Train XGBoost model with weighted combination of all datasets\")"
        ]
    }
]

# Update notebook with new cells
notebook['cells'] = new_cells

# Write updated notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Updated {notebook_path}")
print(f"  - Handles UTM coordinates (Easting/Northing)")
print(f"  - Converts to lat/lon automatically")
print(f"  - Processes Date_Time_MST column")
print(f"  - Includes Feedground analysis")
