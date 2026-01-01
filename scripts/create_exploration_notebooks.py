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

def create_northern_bighorn_notebook():
    """Create notebook for Northern Bighorn migration routes dataset"""
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Exploring North Bighorn Elk Migration Routes\n",
                "\n",
                "This notebook explores the North Bighorn herd migration routes dataset - valuable for Area 048 predictions!\n",
                "\n",
                "**Dataset Info:**\n",
                "- **Location:** Northern Bighorn Mountains, Wyoming\n",
                "- **Data Format:** Shapefile with LineString geometries (migration routes)\n",
                "- **Use Case:** Migration route patterns, seasonal movements, proximity to Area 048\n",
                "- **Note:** North Bighorn herd data complements South Bighorn and other datasets\n",
                "\n",
                "**Data Format:** Shapefile (.shp) with Albers projection - will be converted to WGS84 (lat/lon)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "# Add project root to Python path so we can import src modules\n",
                "# This works whether the notebook is run from project root or notebooks/ directory\n",
                "current_dir = Path().resolve()\n",
                "if current_dir.name == 'notebooks':\n",
                "    project_root = current_dir.parent\n",
                "else:\n",
                "    project_root = current_dir\n",
                "\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.insert(0, str(project_root))\n",
                "\n",
                "import geopandas as gpd\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from shapely.geometry import Point\n",
                "from src.data.hunt_areas import load_area_048_shapefile\n",
                "\n",
                "# Set up paths\n",
                "DATA_DIR = Path(\"../data/raw\")\n",
                "BIGHORN_FILE = DATA_DIR / \"elk_northern_bighorn\" / \"Elk_WY_Bighorn_North_Routes_Ver1_2020.shp\"\n",
                "\n",
                "print(\"Loading North Bighorn migration routes...\")\n",
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
                "    area_048_lat, area_048_lon = 43.4105, -107.5204  # Fallback center"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Load and Inspect the Shapefile"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the shapefile\n",
                "gdf = gpd.read_file(BIGHORN_FILE)\n",
                "\n",
                "print(\"=\" * 60)\n",
                "print(\"NORTH BIGHORN MIGRATION ROUTES DATASET\")\n",
                "print(\"=\" * 60)\n",
                "print(f\"\\nShape: {gdf.shape}\")\n",
                "print(f\"\\nColumns: {list(gdf.columns)}\")\n",
                "print(f\"\\nCRS (Coordinate Reference System): {gdf.crs}\")\n",
                "print(f\"\\nFirst few rows:\")\n",
                "print(gdf.head())\n",
                "print(f\"\\nData types:\")\n",
                "print(gdf.dtypes)\n",
                "print(f\"\\nBasic statistics:\")\n",
                "print(gdf.describe())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 2: Understand the Data Structure"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check what type of geometries we have\n",
                "print(f\"Geometry types: {gdf.geometry.type.unique()}\")\n",
                "print(f\"\\nNumber of routes: {len(gdf)}\")\n",
                "\n",
                "# Check for key columns\n",
                "print(f\"\\nColumn analysis:\")\n",
                "for col in gdf.columns:\n",
                "    if col != 'geometry':\n",
                "        print(f\"  {col}: {gdf[col].dtype}, unique values: {gdf[col].nunique()}\")\n",
                "        if gdf[col].nunique() < 20:\n",
                "            print(f\"    Values: {gdf[col].unique()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Extract Points from Migration Routes"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Migration routes are typically LineString or MultiLineString geometries\n",
                "# We need to extract points along these routes for training\n",
                "\n",
                "def extract_points_from_routes(gdf, points_per_route=100):\n",
                "    \"\"\"Extract sample points from migration route lines\"\"\"\n",
                "    points = []\n",
                "    \n",
                "    for idx, row in gdf.iterrows():\n",
                "        geom = row.geometry\n",
                "        \n",
                "        # Handle LineString\n",
                "        if geom.geom_type == 'LineString':\n",
                "            # Sample points along the line\n",
                "            distances = np.linspace(0, geom.length, points_per_route)\n",
                "            for dist in distances:\n",
                "                point = geom.interpolate(dist)\n",
                "                point_data = row.to_dict()\n",
                "                point_data['geometry'] = point\n",
                "                point_data['route_id'] = idx\n",
                "                point_data['distance_along_route'] = dist\n",
                "                points.append(point_data)\n",
                "        \n",
                "        # Handle MultiLineString\n",
                "        elif geom.geom_type == 'MultiLineString':\n",
                "            for line in geom.geoms:\n",
                "                distances = np.linspace(0, line.length, points_per_route)\n",
                "                for dist in distances:\n",
                "                    point = line.interpolate(dist)\n",
                "                    point_data = row.to_dict()\n",
                "                    point_data['geometry'] = point\n",
                "                    point_data['route_id'] = idx\n",
                "                    point_data['distance_along_route'] = dist\n",
                "                    points.append(point_data)\n",
                "    \n",
                "    return gpd.GeoDataFrame(points, crs=gdf.crs)\n",
                "\n",
                "# Extract points (this may take a moment)\n",
                "print(\"Extracting points from migration routes...\")\n",
                "points_gdf = extract_points_from_routes(gdf, points_per_route=50)\n",
                "\n",
                "print(f\"\\nExtracted {len(points_gdf)} points from {len(gdf)} routes\")\n",
                "print(f\"\\nPoints GeoDataFrame columns: {list(points_gdf.columns)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Convert to Lat/Lon and Analyze Spatial Coverage"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
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
                "    # Convert polygon to UTM once (more accurate for distance calculations)\n",
                "    from pyproj import Transformer\n",
                "    polygon_utm_gdf = gpd.GeoSeries([area_048_polygon], crs='EPSG:4326').to_crs('EPSG:32612')\n",
                "    polygon_utm = polygon_utm_gdf.geometry.iloc[0]\n",
                "    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32612', always_xy=True)\n",
                "    \n",
                "    def distance_to_polygon(point_geom, polygon_wgs84, polygon_utm, trans):\n",
                "        \"\"\"Calculate distance from point to polygon boundary (km)\"\"\"\n",
                "        # Check if point is inside polygon first\n",
                "        if polygon_wgs84.contains(point_geom):\n",
                "            return 0.0\n",
                "        \n",
                "        # Convert point to UTM for accurate distance\n",
                "        point_utm = trans.transform(point_geom.x, point_geom.y)\n",
                "        from shapely.geometry import Point\n",
                "        point_shapely_utm = Point(point_utm[0], point_utm[1])\n",
                "        \n",
                "        # Calculate distance to boundary in meters, convert to km\n",
                "        dist_m = point_shapely_utm.distance(polygon_utm.boundary)\n",
                "        return dist_m / 1000\n",
                "    \n",
                "    # Calculate distance to polygon boundary for each point\n",
                "    points_gdf_wgs84['distance_to_area_048_km'] = points_gdf_wgs84.geometry.apply(\n",
                "        lambda geom: distance_to_polygon(geom, area_048_polygon, polygon_utm, transformer)\n",
                "    )\n",
                "    \n",
                "    # Check which points are inside Area 048\n",
                "    points_gdf_wgs84['inside_area_048'] = points_gdf_wgs84.geometry.apply(\n",
                "        lambda geom: area_048_polygon.contains(geom)\n",
                "    )\n",
                "    \n",
                "    print(f\"\\nProximity to Area 048:\")\n",
                "    print(f\"  Points inside Area 048: {points_gdf_wgs84['inside_area_048'].sum()} ({(points_gdf_wgs84['inside_area_048'].sum() / len(points_gdf_wgs84) * 100):.1f}%)\")\n",
                "    \n",
                "    outside_points = points_gdf_wgs84[~points_gdf_wgs84['inside_area_048']]\n",
                "    if len(outside_points) > 0:\n",
                "        print(f\"  Minimum distance (outside): {outside_points['distance_to_area_048_km'].min():.2f} km\")\n",
                "    print(f\"  Maximum distance: {points_gdf_wgs84['distance_to_area_048_km'].max():.2f} km\")\n",
                "    print(f\"  Average distance: {points_gdf_wgs84['distance_to_area_048_km'].mean():.2f} km\")\n",
                "    points_within_50km = (points_gdf_wgs84['distance_to_area_048_km'] <= 50).sum()\n",
                "    pct_within_50km = points_within_50km / len(points_gdf_wgs84) * 100\n",
                "    print(f\"  Points within 50km: {points_within_50km} ({pct_within_50km:.1f}%)\")\n",
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
                "    print(f\"\\n⚠️  Using center point fallback (polygon not loaded)\")"
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
                "    # Check if we have date columns\n",
                "    date_columns = [col for col in gdf.columns if 'date' in col.lower() or 'time' in col.lower()]\n",
                "    \n",
                "    if date_columns:\n",
                "        for date_col in date_columns:\n",
                "            try:\n",
                "                gdf_wgs84[date_col + '_parsed'] = pd.to_datetime(gdf_wgs84[date_col], errors='coerce')\n",
                "                gdf_wgs84['year'] = gdf_wgs84[date_col + '_parsed'].dt.year\n",
                "                gdf_wgs84['month'] = gdf_wgs84[date_col + '_parsed'].dt.month\n",
                "                \n",
                "                # Map route-level dates to points\n",
                "                route_dates = gdf_wgs84[[date_col + '_parsed', 'year', 'month']].to_dict('index')\n",
                "                for idx, route_data in route_dates.items():\n",
                "                    mask = points_gdf_wgs84['route_id'] == idx\n",
                "                    if date_col + '_parsed' not in points_gdf_wgs84.columns:\n",
                "                        points_gdf_wgs84[date_col + '_parsed'] = None\n",
                "                    if 'year' not in points_gdf_wgs84.columns:\n",
                "                        points_gdf_wgs84['year'] = None\n",
                "                    if 'month' not in points_gdf_wgs84.columns:\n",
                "                        points_gdf_wgs84['month'] = None\n",
                "                    points_gdf_wgs84.loc[mask, date_col + '_parsed'] = route_data[date_col + '_parsed']\n",
                "                    points_gdf_wgs84.loc[mask, 'year'] = route_data['year']\n",
                "                    points_gdf_wgs84.loc[mask, 'month'] = route_data['month']\n",
                "                \n",
                "                print(f\"\\nDate range from {date_col}:\")\n",
                "                valid_dates = gdf_wgs84[date_col + '_parsed'].dropna()\n",
                "                if len(valid_dates) > 0:\n",
                "                    print(f\"  {valid_dates.min()} to {valid_dates.max()}\")\n",
                "                    print(f\"  Total days: {(valid_dates.max() - valid_dates.min()).days}\")\n",
                "                    \n",
                "                    print(f\"\\nYear distribution:\")\n",
                "                    for year, count in gdf_wgs84['year'].value_counts().sort_index().items():\n",
                "                        if pd.notna(year):\n",
                "                            print(f\"  {int(year)}: {count:,} routes ({count/len(gdf_wgs84)*100:.1f}%)\")\n",
                "                    \n",
                "                    print(f\"\\nMonth distribution:\")\n",
                "                    for month, count in gdf_wgs84['month'].value_counts().sort_index().items():\n",
                "                        if pd.notna(month):\n",
                "                            month_name = pd.to_datetime(f\"2020-{int(month)}-01\").strftime(\"%B\")\n",
                "                            print(f\"  {month_name} ({int(month)}): {count:,} routes ({count/len(gdf_wgs84)*100:.1f}%)\")\n",
                "                    \n",
                "                    # Check October data\n",
                "                    october_routes = gdf_wgs84[gdf_wgs84['month'] == 10]\n",
                "                    if len(october_routes) > 0:\n",
                "                        print(f\"\\n🎯 October routes: {len(october_routes):,} ({len(october_routes)/len(gdf_wgs84)*100:.1f}%)\")\n",
                "                    else:\n",
                "                        print(f\"\\n⚠️  No October routes found\")\n",
                "                break\n",
                "            except Exception as e:\n",
                "                print(f\"⚠️  Could not parse {date_col}: {e}\")\n",
                "                continue\n",
                "    else:\n",
                "        print(\"\\n⚠️  No date columns found in dataset\")\n",
                "        print(f\"   Available columns: {list(gdf.columns)}\")\n",
                "    \n",
                "    # Check for season column\n",
                "    if 'season' in gdf.columns:\n",
                "        print(f\"\\nSeason distribution:\")\n",
                "        for season, count in gdf['season'].value_counts().items():\n",
                "            print(f\"  {season}: {count:,} routes ({count/len(gdf)*100:.1f}%)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 6: Visualize Migration Routes"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
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
                "ax.set_title('North Bighorn Elk Migration Routes - Area 048 Boundary', fontsize=16, fontweight='bold')\n",
                "ax.set_xlabel('Longitude', fontsize=12)\n",
                "ax.set_ylabel('Latitude', fontsize=12)\n",
                "ax.legend(fontsize=10)\n",
                "ax.grid(True, alpha=0.3)\n",
                "plt.tight_layout()\n",
                "plt.savefig('../data/processed/north_bighorn_routes_area_048.png', dpi=150, bbox_inches='tight')\n",
                "plt.show()\n",
                "\n",
                "print(\"\\n✓ Map saved to data/processed/north_bighorn_routes_area_048.png\")\n",
                "if area_048_polygon is not None:\n",
                "    print(\"  Red polygon = Area 048 boundary\")\n",
                "    print(\"  Points inside polygon are within Area 048\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 7: Understand Route Attributes"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check what information is available about each route\n",
                "print(\"Route Attributes:\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# Look for columns that might indicate season, direction, etc.\n",
                "for col in gdf.columns:\n",
                "    if col != 'geometry':\n",
                "        print(f\"\\n{col}:\")\n",
                "        print(f\"  Type: {gdf[col].dtype}\")\n",
                "        print(f\"  Unique values: {gdf[col].nunique()}\")\n",
                "        if gdf[col].dtype == 'object':\n",
                "            sample_values = gdf[col].unique()[:5]\n",
                "            print(f\"  Sample values: {sample_values}\")\n",
                "        elif gdf[col].dtype in ['int64', 'float64']:\n",
                "            print(f\"  Range: {gdf[col].min()} to {gdf[col].max()}\")\n",
                "            if gdf[col].nunique() > 1:\n",
                "                print(f\"  Mean: {gdf[col].mean():.2f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 8: Prepare Data for PathWild Integration"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
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
                "# Add temporal info if available\n",
                "if 'year' in points_gdf_wgs84.columns:\n",
                "    pathwild_data['year'] = points_gdf_wgs84['year']\n",
                "if 'month' in points_gdf_wgs84.columns:\n",
                "    pathwild_data['month'] = points_gdf_wgs84['month']\n",
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
                "output_file = Path(\"../data/processed/north_bighorn_points.csv\")\n",
                "output_file.parent.mkdir(parents=True, exist_ok=True)\n",
                "pathwild_data.to_csv(output_file, index=False)\n",
                "print(f\"\\n✓ Saved to {output_file}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 9: Summary and Next Steps"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=\" * 60)\n",
                "print(\"NORTH BIGHORN DATASET SUMMARY\")\n",
                "print(\"=\" * 60)\n",
                "print(f\"\\nTotal migration routes: {len(gdf)}\")\n",
                "print(f\"Total points extracted: {len(points_gdf_wgs84):,}\")\n",
                "print(f\"\\nGeographic coverage:\")\n",
                "print(f\"  Latitude: {points_gdf_wgs84['latitude'].min():.4f}° to {points_gdf_wgs84['latitude'].max():.4f}°\")\n",
                "print(f\"  Longitude: {points_gdf_wgs84['longitude'].min():.4f} to {points_gdf_wgs84['longitude'].max():.4f}°\")\n",
                "print(f\"\\nProximity to Area 048:\")\n",
                "if 'inside_area_048' in points_gdf_wgs84.columns:\n",
                "    inside_count = points_gdf_wgs84['inside_area_048'].sum()\n",
                "    print(f\"  Points inside Area 048: {inside_count:,} ({inside_count / len(points_gdf_wgs84) * 100:.1f}%)\")\n",
                "    points_within_50km = (points_gdf_wgs84['distance_to_area_048_km'] <= 50).sum()\n",
                "    pct_within_50km = points_within_50km / len(points_gdf_wgs84) * 100\n",
                "    print(f\"  Points within 50km: {points_within_50km:,} ({pct_within_50km:.1f}%)\")\n",
                "    if inside_count > 0:\n",
                "        outside_points = points_gdf_wgs84[~points_gdf_wgs84['inside_area_048']]\n",
                "        if len(outside_points) > 0:\n",
                "            print(f\"  Closest point (outside): {outside_points['distance_to_area_048_km'].min():.2f} km away\")\n",
                "    else:\n",
                "        print(f\"  Closest point: {points_gdf_wgs84['distance_to_area_048_km'].min():.2f} km away\")\n",
                "else:\n",
                "    points_within_50km = (points_gdf_wgs84['distance_to_area_048_km'] <= 50).sum()\n",
                "    pct_within_50km = points_within_50km / len(points_gdf_wgs84) * 100\n",
                "    print(f\"  Points within 50km: {points_within_50km:,} ({pct_within_50km:.1f}%)\")\n",
                "    print(f\"  Closest point: {points_gdf_wgs84['distance_to_area_048_km'].min():.2f} km away\")\n",
                "\n",
                "if area_048_polygon is not None:\n",
                "    print(f\"\\nArea 048 Information:\")\n",
                "    print(f\"  Hunt Name: {area_048_gdf['HUNTNAME'].iloc[0]}\")\n",
                "    print(f\"  Size: {area_048_gdf['SqMiles'].iloc[0]:.2f} sq miles\")\n",
                "    if 'HERDNAME' in area_048_gdf.columns:\n",
                "        print(f\"  Herd: {area_048_gdf['HERDNAME'].iloc[0]}\")\n",
                "\n",
                "# Temporal info if available\n",
                "if 'year' in gdf.columns:\n",
                "    print(f\"\\nTemporal coverage:\")\n",
                "    years = sorted([y for y in gdf['year'].dropna().unique() if pd.notna(y)])\n",
                "    if years:\n",
                "        print(f\"  Years: {years}\")\n",
                "    if 'month' in gdf.columns:\n",
                "        months = sorted([m for m in gdf['month'].dropna().unique() if pd.notna(m)])\n",
                "        if months:\n",
                "            print(f\"  Months: {[int(m) for m in months]}\")\n",
                "\n",
                "print(f\"\\n📋 Key Insights:\")\n",
                "print(f\"  ✓ Migration route data from North Bighorn herd\")\n",
                "print(f\"  ✓ {len(points_gdf_wgs84):,} points extracted from {len(gdf)} routes\")\n",
                "if 'inside_area_048' in points_gdf_wgs84.columns:\n",
                "    inside_pct = points_gdf_wgs84['inside_area_048'].sum() / len(points_gdf_wgs84) * 100\n",
                "    print(f\"  ✓ {points_gdf_wgs84['inside_area_048'].sum():,} points ({inside_pct:.1f}%) inside Area 048\")\n",
                "    if points_gdf_wgs84['inside_area_048'].sum() > 0:\n",
                "        print(f\"  → Excellent for Area 048 training data!\")\n",
                "    else:\n",
                "        print(f\"  ⚠️  Routes are outside Area 048 but may still be valuable for training\")\n",
                "\n",
                "print(f\"\\nNext steps:\")\n",
                "print(\"  1. Review the route attributes to understand seasonal patterns\")\n",
                "print(\"  2. Compare with South Bighorn dataset for consistency\")\n",
                "print(\"  3. Integrate with DataContextBuilder to add environmental features\")\n",
                "print(\"  4. Create training dataset with positive examples (route points)\")\n",
                "print(\"  5. Generate negative examples (random points not on routes)\")\n",
                "print(\"  6. Train XGBoost model with your heuristics + ML features\")"
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
    """Create all exploration notebooks"""
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
    
    # Create Northern Bighorn notebook
    bighorn_nb = create_northern_bighorn_notebook()
    bighorn_path = notebooks_dir / "05_explore_northern_bighorn.ipynb"
    with open(bighorn_path, 'w') as f:
        json.dump(bighorn_nb, f, indent=2)
    print(f"✓ Created {bighorn_path}")
    
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
    print("\n3. Ensure Northern Bighorn data is available:")
    print("   → File should be at: data/raw/elk_northern_bighorn/Elk_WY_Bighorn_North_Routes_Ver1_2020.shp")
    print("\n4. Run the notebooks to explore and process the data!")

if __name__ == "__main__":
    main()
