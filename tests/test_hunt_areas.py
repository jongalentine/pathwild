"""
Tests for hunt_areas module.

Tests Wyoming Hunt Area 048 definitions, polygon loading, and point-in-polygon checks.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from shapely.geometry import Point, Polygon, box
import geopandas as gpd
import pandas as pd

from src.data import hunt_areas


class TestArea048Constants:
    """Test Area 048 constant definitions."""
    
    def test_area_048_center(self):
        """Test that center coordinates are correct."""
        center = hunt_areas.get_area_048_center()
        assert center == (43.4105, -107.5204)
        assert isinstance(center, tuple)
        assert len(center) == 2
    
    def test_area_048_constants(self):
        """Test that all constants are defined correctly."""
        assert hunt_areas.AREA_048_NAME == "Area 048"
        assert hunt_areas.AREA_048_FULL_NAME == "Wyoming Hunt Area 048 (Upper Nowood)"
        assert hunt_areas.AREA_048_HUNT_NAME == "Upper Nowood"
        assert hunt_areas.AREA_048_SIZE_SQ_MILES == 930.45
        assert hunt_areas.AREA_048_SIZE_ACRES == 595488.0
        assert hunt_areas.AREA_048_HERD_UNIT == 322.0
        assert hunt_areas.AREA_048_HERD_NAME == "South Bighorn"
        assert hunt_areas.AREA_048_REGION == "Western"
    
    def test_area_048_bbox(self):
        """Test bounding box coordinates."""
        bbox = hunt_areas.AREA_048_BBOX
        assert bbox["min_lat"] == 43.0707
        assert bbox["max_lat"] == 43.7503
        assert bbox["min_lon"] == -107.9076
        assert bbox["max_lon"] == -107.1333
        assert bbox["min_lat"] < bbox["max_lat"]
        assert bbox["min_lon"] < bbox["max_lon"]


class TestGetArea048Polygon:
    """Test get_area_048_polygon() function."""
    
    @patch('src.data.hunt_areas.Path.exists')
    @patch('src.data.hunt_areas.gpd.read_file')
    def test_get_polygon_success(self, mock_read_file, mock_exists):
        """Test successful polygon loading from shapefile."""
        # Mock shapefile exists
        mock_exists.return_value = True
        
        # Create mock polygon
        mock_polygon = box(-107.9, 43.0, -107.1, 43.8)
        mock_gdf = gpd.GeoDataFrame({'geometry': [mock_polygon]})
        mock_read_file.return_value = mock_gdf
        
        result = hunt_areas.get_area_048_polygon()
        
        assert result is not None
        assert isinstance(result, Polygon)
        mock_read_file.assert_called_once()
    
    @patch('src.data.hunt_areas.Path.exists')
    def test_get_polygon_missing_file(self, mock_exists):
        """Test polygon loading when shapefile doesn't exist."""
        mock_exists.return_value = False
        
        result = hunt_areas.get_area_048_polygon()
        
        assert result is None
    
    @patch('src.data.hunt_areas.Path.exists')
    @patch('src.data.hunt_areas.gpd.read_file')
    def test_get_polygon_empty_geodataframe(self, mock_read_file, mock_exists):
        """Test polygon loading when GeoDataFrame is empty."""
        mock_exists.return_value = True
        mock_gdf = gpd.GeoDataFrame()
        mock_read_file.return_value = mock_gdf
        
        result = hunt_areas.get_area_048_polygon()
        
        assert result is None
    
    @patch('src.data.hunt_areas.Path.exists')
    @patch('src.data.hunt_areas.gpd.read_file')
    def test_get_polygon_exception_handling(self, mock_read_file, mock_exists):
        """Test polygon loading when exception occurs."""
        mock_exists.return_value = True
        mock_read_file.side_effect = Exception("File read error")
        
        result = hunt_areas.get_area_048_polygon()
        
        assert result is None


class TestLoadArea048Shapefile:
    """Test load_area_048_shapefile() function."""
    
    @patch('src.data.hunt_areas.gpd.read_file')
    def test_load_shapefile_default_path(self, mock_read_file):
        """Test loading with default path."""
        mock_polygon = box(-107.9, 43.0, -107.1, 43.8)
        mock_gdf = gpd.GeoDataFrame({'geometry': [mock_polygon]})
        mock_read_file.return_value = mock_gdf
        
        result = hunt_areas.load_area_048_shapefile()
        
        assert result is not None
        assert isinstance(result, gpd.GeoDataFrame)
        mock_read_file.assert_called_once()
    
    @patch('src.data.hunt_areas.gpd.read_file')
    def test_load_shapefile_custom_path(self, mock_read_file):
        """Test loading with custom path."""
        custom_path = "/custom/path/Area_048.shp"
        mock_polygon = box(-107.9, 43.0, -107.1, 43.8)
        mock_gdf = gpd.GeoDataFrame({'geometry': [mock_polygon]})
        mock_read_file.return_value = mock_gdf
        
        result = hunt_areas.load_area_048_shapefile(custom_path)
        
        assert result is not None
        mock_read_file.assert_called_once_with(custom_path)
    
    @patch('src.data.hunt_areas.gpd.read_file')
    def test_load_shapefile_with_filtering(self, mock_read_file):
        """Test loading when shapefile contains multiple areas and needs filtering."""
        # Create mock GeoDataFrame with multiple hunt areas
        mock_polygon_048 = box(-107.9, 43.0, -107.1, 43.8)
        mock_polygon_049 = box(-108.0, 44.0, -107.0, 44.8)
        
        mock_gdf = gpd.GeoDataFrame({
            'HUNTAREA': [48.0, 49.0],
            'geometry': [mock_polygon_048, mock_polygon_049]
        })
        mock_read_file.return_value = mock_gdf
        
        result = hunt_areas.load_area_048_shapefile()
        
        assert result is not None
        assert len(result) == 1
        assert result['HUNTAREA'].iloc[0] == 48.0
    
    @patch('src.data.hunt_areas.gpd.read_file')
    def test_load_shapefile_single_area_no_filtering(self, mock_read_file):
        """Test loading when shapefile has single area (no filtering needed)."""
        mock_polygon = box(-107.9, 43.0, -107.1, 43.8)
        mock_gdf = gpd.GeoDataFrame({'geometry': [mock_polygon]})
        mock_read_file.return_value = mock_gdf
        
        result = hunt_areas.load_area_048_shapefile()
        
        assert result is not None
        assert len(result) == 1
    
    @patch('src.data.hunt_areas.gpd.read_file')
    def test_load_shapefile_exception_handling(self, mock_read_file):
        """Test loading when exception occurs."""
        mock_read_file.side_effect = Exception("File read error")
        
        result = hunt_areas.load_area_048_shapefile()
        
        assert result is None


class TestPointInArea048:
    """Test point_in_area_048() function."""
    
    def test_point_in_area_with_polygon(self):
        """Test point-in-polygon check with provided polygon."""
        # Create a simple test polygon (small box around center)
        test_polygon = box(-107.6, 43.3, -107.4, 43.5)
        
        # Point inside polygon
        assert hunt_areas.point_in_area_048(43.4, -107.5, test_polygon) is True
        
        # Point outside polygon
        assert hunt_areas.point_in_area_048(44.0, -108.0, test_polygon) is False
    
    @patch('src.data.hunt_areas.get_area_048_polygon')
    def test_point_in_area_with_loaded_polygon(self, mock_get_polygon):
        """Test point-in-polygon check with polygon loaded from file."""
        test_polygon = box(-107.6, 43.3, -107.4, 43.5)
        mock_get_polygon.return_value = test_polygon
        
        result = hunt_areas.point_in_area_048(43.4, -107.5)
        
        assert result is True
        mock_get_polygon.assert_called_once()
    
    @patch('src.data.hunt_areas.get_area_048_polygon')
    def test_point_in_area_bbox_fallback(self, mock_get_polygon):
        """Test bounding box fallback when polygon is None."""
        mock_get_polygon.return_value = None
        
        bbox = hunt_areas.AREA_048_BBOX
        
        # Point inside bbox
        lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
        lon = (bbox["min_lon"] + bbox["max_lon"]) / 2
        assert hunt_areas.point_in_area_048(lat, lon) is True
        
        # Point outside bbox (too far north)
        assert hunt_areas.point_in_area_048(44.0, lon) is False
        
        # Point outside bbox (too far south)
        assert hunt_areas.point_in_area_048(43.0, lon) is False
        
        # Point outside bbox (too far east)
        assert hunt_areas.point_in_area_048(lat, -107.0) is False
        
        # Point outside bbox (too far west)
        assert hunt_areas.point_in_area_048(lat, -108.0) is False
    
    @patch('src.data.hunt_areas.get_area_048_polygon')
    def test_point_in_area_bbox_boundary_cases(self, mock_get_polygon):
        """Test bounding box boundary conditions."""
        mock_get_polygon.return_value = None
        
        bbox = hunt_areas.AREA_048_BBOX
        
        # Test exact boundaries (inclusive)
        assert hunt_areas.point_in_area_048(bbox["min_lat"], bbox["min_lon"]) is True
        assert hunt_areas.point_in_area_048(bbox["max_lat"], bbox["max_lon"]) is True
        assert hunt_areas.point_in_area_048(bbox["min_lat"], bbox["max_lon"]) is True
        assert hunt_areas.point_in_area_048(bbox["max_lat"], bbox["min_lon"]) is True
        
        # Test just outside boundaries
        assert hunt_areas.point_in_area_048(bbox["min_lat"] - 0.001, bbox["min_lon"]) is False
        assert hunt_areas.point_in_area_048(bbox["max_lat"] + 0.001, bbox["max_lon"]) is False


class TestHuntAreasRegistry:
    """Test HUNT_AREAS registry."""
    
    def test_hunt_areas_registry_structure(self):
        """Test that registry has correct structure."""
        assert "048" in hunt_areas.HUNT_AREAS
        
        area_data = hunt_areas.HUNT_AREAS["048"]
        assert area_data["name"] == hunt_areas.AREA_048_NAME
        assert area_data["full_name"] == hunt_areas.AREA_048_FULL_NAME
        assert area_data["hunt_name"] == hunt_areas.AREA_048_HUNT_NAME
        assert area_data["center"] == hunt_areas.AREA_048_CENTER
        assert area_data["herd_unit"] == hunt_areas.AREA_048_HERD_UNIT
        assert area_data["herd_name"] == hunt_areas.AREA_048_HERD_NAME
        assert area_data["region"] == hunt_areas.AREA_048_REGION
        assert area_data["size_sq_miles"] == hunt_areas.AREA_048_SIZE_SQ_MILES
        assert area_data["size_acres"] == hunt_areas.AREA_048_SIZE_ACRES
        assert area_data["bbox"] == hunt_areas.AREA_048_BBOX
        assert area_data["shapefile"] == "data/raw/hunt_areas/Area_048.shp"
    
    def test_hunt_areas_registry_all_fields(self):
        """Test that registry entry has all expected fields."""
        area_data = hunt_areas.HUNT_AREAS["048"]
        expected_fields = [
            "name", "full_name", "hunt_name", "center", "herd_unit",
            "herd_name", "region", "size_sq_miles", "size_acres", "bbox", "shapefile"
        ]
        
        for field in expected_fields:
            assert field in area_data, f"Missing field: {field}"

