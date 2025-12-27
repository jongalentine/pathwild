#!/usr/bin/env python3
"""
Script to create exploration notebooks for National Elk Refuge and Southern GYE datasets.
This generates complete notebooks similar to the South Bighorn exploration notebook.
"""

import json
from pathlib import Path

# Common notebook metadata
METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

def create_national_refuge_notebook():
    """Create notebook for National Elk Refuge dataset"""
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exploring National Elk Refuge GPS Collar Data (2006-2015)\n",
                "\n",
                "This notebook explores the National Elk Refuge GPS collar dataset - valuable for general elk behavior patterns and large sample size training!\n",
                "\n",
                "**Dataset Info:**\n",
                "- **Location:** National Elk Refuge, Jackson, Wyoming\n",
                "- **Coverage:** 17 adult female elk, 2006-2015\n",
                "- **Data:** GPS locations, timestamps, migration patterns\n",
                "- **Use Case:** General elk behavior patterns, seasonal timing, long time series\n",
                "- **Note:** ~200 miles from Area 048, but provides valuable general patterns\n",
                "\n",
                "**Download:** https://data.usgs.gov/datacatalog/data/USGS:5a9f2782e4b0b1c392e502ea"
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
                "\n",
                "# Set up paths\n",
                "DATA_DIR = Path(\"../data/raw\")\n",
                "REFUGE_DIR = DATA_DIR / \"elk_national_refuge\"\n",
                "\n",
                "print(\"=\" * 60)\n",
                "print(\"NATIONAL ELK REFUGE DATASET\")\n",
                "print(\"=\" * 60)\n",
                "print(f\"\\nData directory: {REFUGE_DIR}\")\n",
                "print(f\"Directory exists: {REFUGE_DIR.exists()}\")\n",
                "\n",
                "# Look for data files\n",
                "if REFUGE_DIR.exists():\n",
                "    files = list(REFUGE_DIR.glob(\"*\"))\n",
                "    print(f\"\\nFiles found: {len(files)}\")\n",
                "    for f in files[:10]:\n",
                "        print(f\"  - {f.name}\")\n",
                "else:\n",
                "    print(\"\\n⚠️  Directory doesn't exist yet!\")\n",
                "    print(\"📥 Download instructions:\")\n",
                "    print(\"   1. Visit: https://data.usgs.gov/datacatalog/data/USGS:5a9f2782e4b0b1c392e502ea\")\n",
                "    print(\"   2. Download the dataset (CSV or shapefile format)\")\n",
                "    print(\"   3. Extract to: data/raw/elk_national_refuge/\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Load the Data\n",
                "\n",
                "The dataset may be in CSV format (GPS points) or shapefile format. We'll try both."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Try to find and load the data file\n",
                "csv_files = list(REFUGE_DIR.glob(\"*.csv\"))\n",
                "shp_files = list(REFUGE_DIR.glob(\"*.shp\"))\n",
                "\n",
                "if shp_files:\n",
                "    print(f\"Loading shapefile: {shp_files[0].name}\")\n",
                "    gdf = gpd.read_file(shp_files[0])\n",
                "    data_type = \"shapefile\"\n",
                "elif csv_files:\n",
                "    print(f\"Loading CSV: {csv_files[0].name}\")\n",
                "    df = pd.read_csv(csv_files[0])\n",
                "    \n",
                "    # Auto-detect lat/lon columns\n",
                "    lat_col = None\n",
                "    lon_col = None\n",
                "    for col in df.columns:\n",
                "        col_lower = col.lower()\n",
                "        if 'lat' in col_lower and lat_col is None:\n",
                "            lat_col = col\n",
                "        if ('lon' in col_lower or 'long' in col_lower) and lon_col is None:\n",
                "            lon_col = col\n",
                "    \n",
                "    if lat_col and lon_col:\n",
                "        print(f\"  Found coordinates: {lat_col}, {lon_col}\")\n",
                "        gdf = gpd.GeoDataFrame(\n",
                "            df,\n",
                "            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),\n",
                "            crs='EPSG:4326'\n",
                "        )\n",
                "        data_type = \"csv_points\"\n",
                "    else:\n",
                "        print(f\"  ⚠️  Columns: {list(df.columns)}\")\n",
                "        print(\"  Please update the notebook to specify lat/lon column names.\")\n",
                "        gdf = None\n",
                "        data_type = None\n",
                "else:\n",
                "    print(\"⚠️  No data files found!\")\n",
                "    gdf = None\n",
                "    data_type = None\n",
                "\n",
                "if gdf is not None:\n",
                "    print(f\"\\n✓ Data loaded: {data_type}, Shape: {gdf.shape}, CRS: {gdf.crs}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Inspect Dataset Structure"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf is not None:\n",
                "    print(\"=\" * 60)\n",
                "    print(\"DATASET STRUCTURE\")\n",
                "    print(\"=\" * 60)\n",
                "    print(f\"\\nShape: {gdf.shape}\")\n",
                "    print(f\"Columns: {list(gdf.columns)}\")\n",
                "    print(f\"\\nFirst few rows:\")\n",
                "    print(gdf.head())\n",
                "    print(f\"\\nData types:\")\n",
                "    print(gdf.dtypes)\n",
                "    print(f\"\\nMissing values:\")\n",
                "    missing = gdf.isnull().sum()\n",
                "    if missing.sum() > 0:\n",
                "        for col, count in missing[missing > 0].items():\n",
                "            print(f\"  {col}: {count} ({count/len(gdf)*100:.1f}%)\")\n",
                "    else:\n",
                "        print(\"  ✓ No missing values!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Extract Coordinates and Analyze Spatial Coverage"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf is not None:\n",
                "    # Ensure we have lat/lon\n",
                "    if 'latitude' not in gdf.columns or 'longitude' not in gdf.columns:\n",
                "        if gdf.geometry is not None:\n",
                "            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs != 'EPSG:4326' else gdf\n",
                "            gdf_wgs84['latitude'] = gdf_wgs84.geometry.y\n",
                "            gdf_wgs84['longitude'] = gdf_wgs84.geometry.x\n",
                "        else:\n",
                "            gdf_wgs84 = None\n",
                "    else:\n",
                "        gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs != 'EPSG:4326' else gdf\n",
                "    \n",
                "    if gdf_wgs84 is not None:\n",
                "        print(\"=\" * 60)\n",
                "        print(\"SPATIAL COVERAGE\")\n",
                "        print(\"=\" * 60)\n",
                "        print(f\"\\nLatitude: {gdf_wgs84['latitude'].min():.4f}° to {gdf_wgs84['latitude'].max():.4f}°\")\n",
                "        print(f\"Longitude: {gdf_wgs84['longitude'].min():.4f}° to {gdf_wgs84['longitude'].max():.4f}°\")\n",
                "        \n",
                "        # Distance to Area 048\n",
                "        area_048_lat, area_048_lon = 41.835, -106.425\n",
                "        \n",
                "        from math import radians, sin, cos, sqrt, atan2\n",
                "        \n",
                "        def haversine_distance(lat1, lon1, lat2, lon2):\n",
                "            R = 6371  # Earth radius in km\n",
                "            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])\n",
                "            dlat = lat2 - lat1\n",
                "            dlon = lon2 - lon1\n",
                "            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2\n",
                "            c = 2 * atan2(sqrt(a), sqrt(1-a))\n",
                "            return R * c\n",
                "        \n",
                "        gdf_wgs84['distance_to_area_048_km'] = gdf_wgs84.apply(\n",
                "            lambda row: haversine_distance(row['latitude'], row['longitude'], area_048_lat, area_048_lon),\n",
                "            axis=1\n",
                "        )\n",
                "        \n",
                "        print(f\"\\nProximity to Area 048:\")\n",
                "        print(f\"  Min distance: {gdf_wgs84['distance_to_area_048_km'].min():.2f} km\")\n",
                "        print(f\"  Max distance: {gdf_wgs84['distance_to_area_048_km'].max():.2f} km\")\n",
                "        print(f\"  Avg distance: {gdf_wgs84['distance_to_area_048_km'].mean():.2f} km\")\n",
                "        print(f\"  Points within 200km: {(gdf_wgs84['distance_to_area_048_km'] <= 200).sum()} ({(gdf_wgs84['distance_to_area_048_km'] <= 200).sum() / len(gdf_wgs84) * 100:.1f}%)\")\n",
                "        print(f\"\\n⚠️  Note: National Elk Refuge is ~200 miles from Area 048.\")\n",
                "        print(f\"   This data is valuable for general patterns, not geographic specificity.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Analyze Temporal Patterns"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf_wgs84 is not None:\n",
                "    # Try to find date column\n",
                "    date_col = None\n",
                "    for col in gdf_wgs84.columns:\n",
                "        if 'date' in col.lower() or 'time' in col.lower():\n",
                "            date_col = col\n",
                "            break\n",
                "    \n",
                "    if date_col:\n",
                "        try:\n",
                "            gdf_wgs84['date'] = pd.to_datetime(gdf_wgs84[date_col])\n",
                "            gdf_wgs84['year'] = gdf_wgs84['date'].dt.year\n",
                "            gdf_wgs84['month'] = gdf_wgs84['date'].dt.month\n",
                "            \n",
                "            print(\"=\" * 60)\n",
                "            print(\"TEMPORAL ANALYSIS\")\n",
                "            print(\"=\" * 60)\n",
                "            print(f\"\\nDate range: {gdf_wgs84['date'].min()} to {gdf_wgs84['date'].max()}\")\n",
                "            \n",
                "            print(f\"\\nYear distribution:\")\n",
                "            for year, count in gdf_wgs84['year'].value_counts().sort_index().items():\n",
                "                print(f\"  {int(year)}: {count:,} points ({count/len(gdf_wgs84)*100:.1f}%)\")\n",
                "            \n",
                "            print(f\"\\nMonth distribution:\")\n",
                "            for month, count in gdf_wgs84['month'].value_counts().sort_index().items():\n",
                "                month_name = pd.to_datetime(f\"2020-{month}-01\").strftime(\"%B\")\n",
                "                print(f\"  {month_name}: {count:,} points ({count/len(gdf_wgs84)*100:.1f}%)\")\n",
                "            \n",
                "            # October analysis\n",
                "            october_points = gdf_wgs84[gdf_wgs84['month'] == 10]\n",
                "            print(f\"\\n🎯 October data: {len(october_points):,} points ({len(october_points)/len(gdf_wgs84)*100:.1f}%)\")\n",
                "        except Exception as e:\n",
                "            print(f\"⚠️  Could not parse dates: {e}\")\n",
                "    else:\n",
                "        print(\"⚠️  No date column found\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Prepare Data for PathWild Integration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf_wgs84 is not None:\n",
                "    # Create PathWild-ready dataset\n",
                "    pathwild_data = pd.DataFrame({\n",
                "        'latitude': gdf_wgs84['latitude'],\n",
                "        'longitude': gdf_wgs84['longitude'],\n",
                "        'distance_to_area_048_km': gdf_wgs84['distance_to_area_048_km']\n",
                "    })\n",
                "    \n",
                "    # Add temporal info if available\n",
                "    if 'date' in gdf_wgs84.columns:\n",
                "        pathwild_data['date'] = gdf_wgs84['date']\n",
                "        pathwild_data['year'] = gdf_wgs84['year']\n",
                "        pathwild_data['month'] = gdf_wgs84['month']\n",
                "    \n",
                "    # Add other relevant columns\n",
                "    for col in gdf_wgs84.columns:\n",
                "        if col not in pathwild_data.columns and col not in ['geometry', 'latitude', 'longitude']:\n",
                "            if gdf_wgs84[col].dtype in ['int64', 'float64', 'object']:\n",
                "                pathwild_data[col] = gdf_wgs84[col]\n",
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
                "    output_file = Path(\"../data/processed/national_refuge_points.csv\")\n",
                "    output_file.parent.mkdir(parents=True, exist_ok=True)\n",
                "    pathwild_data.to_csv(output_file, index=False)\n",
                "    print(f\"\\n✓ Saved to {output_file}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Summary and Next Steps"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf_wgs84 is not None:\n",
                "    print(\"=\" * 60)\n",
                "    print(\"NATIONAL ELK REFUGE DATASET SUMMARY\")\n",
                "    print(\"=\" * 60)\n",
                "    print(f\"\\nTotal GPS points: {len(gdf_wgs84):,}\")\n",
                "    print(f\"\\nGeographic coverage:\")\n",
                "    print(f\"  Latitude: {gdf_wgs84['latitude'].min():.4f}° to {gdf_wgs84['latitude'].max():.4f}°\")\n",
                "    print(f\"  Longitude: {gdf_wgs84['longitude'].min():.4f}° to {gdf_wgs84['longitude'].max():.4f}°\")\n",
                "    print(f\"\\nProximity to Area 048:\")\n",
                "    print(f\"  Average distance: {gdf_wgs84['distance_to_area_048_km'].mean():.2f} km\")\n",
                "    \n",
                "    print(f\"\\n📋 Key Insights:\")\n",
                "    print(f\"  ✓ Large sample size for general elk behavior patterns\")\n",
                "    print(f\"  ✓ Long time series (2006-2015)\")\n",
                "    print(f\"  ✓ Useful for understanding seasonal timing\")\n",
                "    print(f\"  ⚠️  Geographic distance from Area 048 (~200 miles)\")\n",
                "    print(f\"  → Best used for general patterns, not geographic specificity\")\n",
                "    \n",
                "    print(f\"\\nNext steps:\")\n",
                "    print(\"  1. Combine with South Bighorn data for hybrid training\")\n",
                "    print(\"  2. Use for general elk behavior patterns\")\n",
                "    print(\"  3. Integrate with DataContextBuilder to add environmental features\")\n",
                "    print(\"  4. Create training dataset with positive examples (GPS points)\")\n",
                "    print(\"  5. Generate negative examples (random points)\")\n",
                "    print(\"  6. Train XGBoost model with weighted combination of datasets\")"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def create_southern_gye_notebook():
    """Create notebook for Southern GYE dataset"""
    
    cells = [
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
                "**Download:** https://catalog.data.gov/dataset/elk-gps-collar-data-in-southern-gye-2007-2015"
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
                "    files = list(GYE_DIR.glob(\"*\"))\n",
                "    print(f\"\\nFiles found: {len(files)}\")\n",
                "    for f in files[:10]:\n",
                "        print(f\"  - {f.name}\")\n",
                "else:\n",
                "    print(\"\\n⚠️  Directory doesn't exist yet!\")\n",
                "    print(\"📥 Download instructions:\")\n",
                "    print(\"   1. Visit: https://catalog.data.gov/dataset/elk-gps-collar-data-in-southern-gye-2007-2015\")\n",
                "    print(\"   2. Download the dataset (CSV format)\")\n",
                "    print(\"   3. Extract to: data/raw/elk_southern_gye/\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Load the Data\n",
                "\n",
                "The dataset is typically in CSV format with GPS points."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Try to find and load the data file\n",
                "csv_files = list(GYE_DIR.glob(\"*.csv\"))\n",
                "shp_files = list(GYE_DIR.glob(\"*.shp\"))\n",
                "\n",
                "if shp_files:\n",
                "    print(f\"Loading shapefile: {shp_files[0].name}\")\n",
                "    gdf = gpd.read_file(shp_files[0])\n",
                "    data_type = \"shapefile\"\n",
                "elif csv_files:\n",
                "    print(f\"Loading CSV: {csv_files[0].name}\")\n",
                "    df = pd.read_csv(csv_files[0])\n",
                "    \n",
                "    # Auto-detect lat/lon columns\n",
                "    lat_col = None\n",
                "    lon_col = None\n",
                "    for col in df.columns:\n",
                "        col_lower = col.lower()\n",
                "        if 'lat' in col_lower and lat_col is None:\n",
                "            lat_col = col\n",
                "        if ('lon' in col_lower or 'long' in col_lower) and lon_col is None:\n",
                "            lon_col = col\n",
                "    \n",
                "    if lat_col and lon_col:\n",
                "        print(f\"  Found coordinates: {lat_col}, {lon_col}\")\n",
                "        gdf = gpd.GeoDataFrame(\n",
                "            df,\n",
                "            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),\n",
                "            crs='EPSG:4326'\n",
                "        )\n",
                "        data_type = \"csv_points\"\n",
                "    else:\n",
                "        print(f\"  ⚠️  Columns: {list(df.columns)}\")\n",
                "        print(\"  Please update the notebook to specify lat/lon column names.\")\n",
                "        gdf = None\n",
                "        data_type = None\n",
                "else:\n",
                "    print(\"⚠️  No data files found!\")\n",
                "    gdf = None\n",
                "    data_type = None\n",
                "\n",
                "if gdf is not None:\n",
                "    print(f\"\\n✓ Data loaded: {data_type}, Shape: {gdf.shape}, CRS: {gdf.crs}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Inspect Dataset Structure"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf is not None:\n",
                "    print(\"=\" * 60)\n",
                "    print(\"DATASET STRUCTURE\")\n",
                "    print(\"=\" * 60)\n",
                "    print(f\"\\nShape: {gdf.shape}\")\n",
                "    print(f\"Columns: {list(gdf.columns)}\")\n",
                "    print(f\"\\nFirst few rows:\")\n",
                "    print(gdf.head())\n",
                "    print(f\"\\nData types:\")\n",
                "    print(gdf.dtypes)\n",
                "    print(f\"\\nMissing values:\")\n",
                "    missing = gdf.isnull().sum()\n",
                "    if missing.sum() > 0:\n",
                "        for col, count in missing[missing > 0].items():\n",
                "            print(f\"  {col}: {count} ({count/len(gdf)*100:.1f}%)\")\n",
                "    else:\n",
                "        print(\"  ✓ No missing values!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Extract Coordinates and Analyze Spatial Coverage"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf is not None:\n",
                "    # Ensure we have lat/lon\n",
                "    if 'latitude' not in gdf.columns or 'longitude' not in gdf.columns:\n",
                "        if gdf.geometry is not None:\n",
                "            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs != 'EPSG:4326' else gdf\n",
                "            gdf_wgs84['latitude'] = gdf_wgs84.geometry.y\n",
                "            gdf_wgs84['longitude'] = gdf_wgs84.geometry.x\n",
                "        else:\n",
                "            gdf_wgs84 = None\n",
                "    else:\n",
                "        gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs != 'EPSG:4326' else gdf\n",
                "    \n",
                "    if gdf_wgs84 is not None:\n",
                "        print(\"=\" * 60)\n",
                "        print(\"SPATIAL COVERAGE\")\n",
                "        print(\"=\" * 60)\n",
                "        print(f\"\\nLatitude: {gdf_wgs84['latitude'].min():.4f}° to {gdf_wgs84['latitude'].max():.4f}°\")\n",
                "        print(f\"Longitude: {gdf_wgs84['longitude'].min():.4f}° to {gdf_wgs84['longitude'].max():.4f}°\")\n",
                "        \n",
                "        # Distance to Area 048\n",
                "        area_048_lat, area_048_lon = 41.835, -106.425\n",
                "        \n",
                "        from math import radians, sin, cos, sqrt, atan2\n",
                "        \n",
                "        def haversine_distance(lat1, lon1, lat2, lon2):\n",
                "            R = 6371  # Earth radius in km\n",
                "            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])\n",
                "            dlat = lat2 - lat1\n",
                "            dlon = lon2 - lon1\n",
                "            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2\n",
                "            c = 2 * atan2(sqrt(a), sqrt(1-a))\n",
                "            return R * c\n",
                "        \n",
                "        gdf_wgs84['distance_to_area_048_km'] = gdf_wgs84.apply(\n",
                "            lambda row: haversine_distance(row['latitude'], row['longitude'], area_048_lat, area_048_lon),\n",
                "            axis=1\n",
                "        )\n",
                "        \n",
                "        print(f\"\\nProximity to Area 048:\")\n",
                "        print(f\"  Min distance: {gdf_wgs84['distance_to_area_048_km'].min():.2f} km\")\n",
                "        print(f\"  Max distance: {gdf_wgs84['distance_to_area_048_km'].max():.2f} km\")\n",
                "        print(f\"  Avg distance: {gdf_wgs84['distance_to_area_048_km'].mean():.2f} km\")\n",
                "        print(f\"  Points within 200km: {(gdf_wgs84['distance_to_area_048_km'] <= 200).sum()} ({(gdf_wgs84['distance_to_area_048_km'] <= 200).sum() / len(gdf_wgs84) * 100:.1f}%)\")\n",
                "        print(f\"\\n⚠️  Note: Southern GYE is ~200 miles from Area 048.\")\n",
                "        print(f\"   This data is valuable for large sample size training.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Analyze Temporal Patterns"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf_wgs84 is not None:\n",
                "    # Try to find date column\n",
                "    date_col = None\n",
                "    for col in gdf_wgs84.columns:\n",
                "        if 'date' in col.lower() or 'time' in col.lower():\n",
                "            date_col = col\n",
                "            break\n",
                "    \n",
                "    if date_col:\n",
                "        try:\n",
                "            gdf_wgs84['date'] = pd.to_datetime(gdf_wgs84[date_col])\n",
                "            gdf_wgs84['year'] = gdf_wgs84['date'].dt.year\n",
                "            gdf_wgs84['month'] = gdf_wgs84['date'].dt.month\n",
                "            \n",
                "            print(\"=\" * 60)\n",
                "            print(\"TEMPORAL ANALYSIS\")\n",
                "            print(\"=\" * 60)\n",
                "            print(f\"\\nDate range: {gdf_wgs84['date'].min()} to {gdf_wgs84['date'].max()}\")\n",
                "            \n",
                "            print(f\"\\nYear distribution:\")\n",
                "            for year, count in gdf_wgs84['year'].value_counts().sort_index().items():\n",
                "                print(f\"  {int(year)}: {count:,} points ({count/len(gdf_wgs84)*100:.1f}%)\")\n",
                "            \n",
                "            print(f\"\\nMonth distribution:\")\n",
                "            for month, count in gdf_wgs84['month'].value_counts().sort_index().items():\n",
                "                month_name = pd.to_datetime(f\"2020-{month}-01\").strftime(\"%B\")\n",
                "                print(f\"  {month_name}: {count:,} points ({count/len(gdf_wgs84)*100:.1f}%)\")\n",
                "            \n",
                "            # Note: Data is Feb-July (brucellosis risk period)\n",
                "            print(f\"\\n📋 Note: This dataset focuses on Feb-July (brucellosis risk period)\")\n",
                "            print(f\"   October data may be limited, but still valuable for general patterns.\")\n",
                "        except Exception as e:\n",
                "            print(f\"⚠️  Could not parse dates: {e}\")\n",
                "    else:\n",
                "        print(\"⚠️  No date column found\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Analyze Elk Individual Patterns"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf_wgs84 is not None:\n",
                "    # Try to find elk ID column\n",
                "    elk_id_col = None\n",
                "    for col in gdf_wgs84.columns:\n",
                "        if 'id' in col.lower() or 'elk' in col.lower():\n",
                "            elk_id_col = col\n",
                "            break\n",
                "    \n",
                "    if elk_id_col:\n",
                "        print(\"=\" * 60)\n",
                "        print(\"ELK INDIVIDUAL ANALYSIS\")\n",
                "        print(\"=\" * 60)\n",
                "        print(f\"\\nTotal unique elk: {gdf_wgs84[elk_id_col].nunique()}\")\n",
                "        print(f\"Total GPS points: {len(gdf_wgs84):,}\")\n",
                "        print(f\"Average points per elk: {len(gdf_wgs84) / gdf_wgs84[elk_id_col].nunique():.0f}\")\n",
                "        \n",
                "        points_per_elk = gdf_wgs84[elk_id_col].value_counts()\n",
                "        print(f\"\\nPoints per elk:\")\n",
                "        print(f\"  Minimum: {points_per_elk.min():,}\")\n",
                "        print(f\"  Maximum: {points_per_elk.max():,}\")\n",
                "        print(f\"  Mean: {points_per_elk.mean():.0f}\")\n",
                "        print(f\"  Median: {points_per_elk.median():.0f}\")\n",
                "    else:\n",
                "        print(\"⚠️  No elk ID column found\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Prepare Data for PathWild Integration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf_wgs84 is not None:\n",
                "    # Create PathWild-ready dataset\n",
                "    pathwild_data = pd.DataFrame({\n",
                "        'latitude': gdf_wgs84['latitude'],\n",
                "        'longitude': gdf_wgs84['longitude'],\n",
                "        'distance_to_area_048_km': gdf_wgs84['distance_to_area_048_km']\n",
                "    })\n",
                "    \n",
                "    # Add temporal info if available\n",
                "    if 'date' in gdf_wgs84.columns:\n",
                "        pathwild_data['date'] = gdf_wgs84['date']\n",
                "        pathwild_data['year'] = gdf_wgs84['year']\n",
                "        pathwild_data['month'] = gdf_wgs84['month']\n",
                "    \n",
                "    # Add elk ID if available\n",
                "    if elk_id_col:\n",
                "        pathwild_data['elk_id'] = gdf_wgs84[elk_id_col]\n",
                "    \n",
                "    # Add other relevant columns\n",
                "    for col in gdf_wgs84.columns:\n",
                "        if col not in pathwild_data.columns and col not in ['geometry', 'latitude', 'longitude']:\n",
                "            if gdf_wgs84[col].dtype in ['int64', 'float64', 'object']:\n",
                "                pathwild_data[col] = gdf_wgs84[col]\n",
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
                "## Step 7: Summary and Next Steps"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if gdf_wgs84 is not None:\n",
                "    print(\"=\" * 60)\n",
                "    print(\"SOUTHERN GYE DATASET SUMMARY\")\n",
                "    print(\"=\" * 60)\n",
                "    print(f\"\\nTotal GPS points: {len(gdf_wgs84):,}\")\n",
                "    if elk_id_col:\n",
                "        print(f\"Unique elk: {gdf_wgs84[elk_id_col].nunique()}\")\n",
                "    print(f\"\\nGeographic coverage:\")\n",
                "    print(f\"  Latitude: {gdf_wgs84['latitude'].min():.4f}° to {gdf_wgs84['latitude'].max():.4f}°\")\n",
                "    print(f\"  Longitude: {gdf_wgs84['longitude'].min():.4f}° to {gdf_wgs84['longitude'].max():.4f}°\")\n",
                "    print(f\"\\nProximity to Area 048:\")\n",
                "    print(f\"  Average distance: {gdf_wgs84['distance_to_area_048_km'].mean():.2f} km\")\n",
                "    \n",
                "    print(f\"\\n📋 Key Insights:\")\n",
                "    print(f\"  ✓ LARGEST sample size (288 elk)\")\n",
                "    print(f\"  ✓ Excellent for statistical robustness\")\n",
                "    print(f\"  ✓ Diverse conditions across 22 feedgrounds\")\n",
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
    
    notebook = {
        "cells": cells,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def main():
    """Create both notebooks"""
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)
    
    # Create National Elk Refuge notebook
    refuge_nb = create_national_refuge_notebook()
    refuge_path = notebooks_dir / "03_explore_national_refuge.ipynb"
    with open(refuge_path, 'w') as f:
        json.dump(refuge_nb, f, indent=2)
    print(f"✓ Created {refuge_path}")
    
    # Create Southern GYE notebook
    gye_nb = create_southern_gye_notebook()
    gye_path = notebooks_dir / "04_explore_southern_gye.ipynb"
    with open(gye_path, 'w') as f:
        json.dump(gye_nb, f, indent=2)
    print(f"✓ Created {gye_path}")
    
    print("\n" + "=" * 60)
    print("NOTEBOOKS CREATED SUCCESSFULLY")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download National Elk Refuge data:")
    print("   https://data.usgs.gov/datacatalog/data/USGS:5a9f2782e4b0b1c392e502ea")
    print("   → Extract to: data/raw/elk_national_refuge/")
    print("\n2. Download Southern GYE data:")
    print("   https://catalog.data.gov/dataset/elk-gps-collar-data-in-southern-gye-2007-2015")
    print("   → Extract to: data/raw/elk_southern_gye/")
    print("\n3. Run the notebooks to explore and process the data!")

if __name__ == "__main__":
    main()
