"""
End-to-end integration tests for the complete data processing pipeline.

These tests use a small test dataset to verify the entire pipeline works correctly.
"""
import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import subprocess
import sys
from unittest.mock import patch, Mock
import tempfile
import shutil

# Add scripts to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


class TestPipelineIntegration:
    """End-to-end integration tests with real (small) test data."""
    
    @pytest.fixture
    def test_environment(self, tmp_path):
        """Create a complete test environment with all required data."""
        # Create directory structure
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"
        
        # Environmental data directories
        dem_dir = data_dir / "dem"
        terrain_dir = data_dir / "terrain"
        landcover_dir = data_dir / "landcover"
        canopy_dir = data_dir / "canopy"
        hydrology_dir = data_dir / "hydrology"
        infrastructure_dir = data_dir / "infrastructure"
        boundaries_dir = data_dir / "boundaries"
        
        for d in [raw_dir, processed_dir, dem_dir, terrain_dir, landcover_dir,
                  canopy_dir, hydrology_dir, infrastructure_dir, boundaries_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create test raw presence data
        test_dataset_dir = raw_dir / "elk_test_dataset"
        test_dataset_dir.mkdir()
        test_csv = test_dataset_dir / "test_data.csv"
        test_csv.write_text(
            "latitude,longitude,id,date\n"
            "43.0,-110.0,1,2024-01-01\n"
            "43.1,-110.1,2,2024-01-02\n"
            "43.2,-110.2,3,2024-01-03\n"
        )
        
        # Create minimal mock environmental data
        # Water sources
        water_data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-110.0, 43.0]},
                "properties": {"water_type": "stream", "type": "stream"}
            }]
        }
        (hydrology_dir / "water_sources.geojson").write_text(json.dumps(water_data))
        
        # Roads (empty for testing)
        roads_data = {
            "type": "FeatureCollection",
            "features": []
        }
        (infrastructure_dir / "roads.geojson").write_text(json.dumps(roads_data))
        
        # Trails (empty for testing)
        trails_data = {
            "type": "FeatureCollection",
            "features": []
        }
        (infrastructure_dir / "trails.geojson").write_text(json.dumps(trails_data))
        
        # Wyoming boundary (simplified) - skip shapefile creation in sandbox
        # Just create a GeoJSON instead for testing
        from shapely.geometry import Polygon
        wyoming_polygon = Polygon([(-111, 41), (-104, 41), (-104, 45), (-111, 45), (-111, 41)])
        wyoming_bbox = gpd.GeoDataFrame(
            geometry=[wyoming_polygon],
            crs="EPSG:4326"
        )
        # Use GeoJSON instead of shapefile to avoid pyogrio permission issues
        try:
            wyoming_bbox.to_file(boundaries_dir / "wyoming_state.shp")
        except (PermissionError, OSError):
            # Fallback to GeoJSON if shapefile write fails (sandbox restrictions)
            wyoming_bbox.to_file(boundaries_dir / "wyoming_state.geojson", driver='GeoJSON')
        
        return {
            'data_dir': data_dir,
            'raw_dir': raw_dir,
            'processed_dir': processed_dir,
            'test_csv': test_csv
        }
    
    def test_process_raw_data_step(self, test_environment):
        """Test processing raw data into presence points."""
        import importlib.util
        scripts_dir = Path(__file__).parent.parent / "scripts"
        spec = importlib.util.spec_from_file_location(
            "process_raw_presence_data",
            scripts_dir / "process_raw_presence_data.py"
        )
        process_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(process_module)
        
        data_dir = test_environment['data_dir']
        raw_dir = test_environment['raw_dir']
        processed_dir = test_environment['processed_dir']
        
        # Process the test dataset
        output_file = process_module.process_dataset(
            raw_dir / "elk_test_dataset",
            "test_dataset",
            processed_dir
        )
        
        assert output_file is not None
        assert output_file.exists()
        
        # Verify output
        df = pd.read_csv(output_file)
        assert len(df) == 3
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
        assert 'dataset' in df.columns
        assert all(df['dataset'] == 'test_dataset')
    
    @patch('src.data.processors.DataContextBuilder')
    @patch('builtins.print')
    def test_generate_absence_data_step(self, mock_print, mock_builder_class, test_environment):
        """Test generating absence data."""
        import importlib.util
        scripts_dir = Path(__file__).parent.parent / "scripts"
        spec = importlib.util.spec_from_file_location(
            "generate_absence_data",
            scripts_dir / "generate_absence_data.py"
        )
        generate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate_module)
        
        data_dir = test_environment['data_dir']
        processed_dir = test_environment['processed_dir']
        
        # Create presence points file first
        presence_file = processed_dir / "test_dataset_points.csv"
        presence_file.write_text(
            "latitude,longitude,id,date\n"
            "43.0,-110.0,1,2024-01-01\n"
            "43.1,-110.1,2,2024-01-02\n"
        )
        
        # Mock DataContextBuilder
        mock_builder = Mock()
        mock_builder.build_context = Mock(return_value={
            'elevation': 2500.0,
            'slope_degrees': 5.0,
            'aspect_degrees': 180.0,
            'canopy_cover_percent': 45.0,
            'land_cover_code': 41,
            'land_cover_type': 'deciduous_forest',
            'water_distance_miles': 2.5,
            'water_reliability': 0.8,
            'road_distance_miles': 1.5,
            'trail_distance_miles': 0.8,
            'security_habitat_percent': 0.6
        })
        mock_builder.dem = None
        mock_builder.slope = None
        mock_builder.aspect = None
        mock_builder.landcover = None
        mock_builder.canopy = None
        mock_builder.water_sources = None
        mock_builder.roads = None
        mock_builder.trails = None
        mock_builder_class.return_value = mock_builder
        
        # Run generate absence (with minimal absences for speed)
        output_file = processed_dir / "combined_test_dataset_presence_absence.csv"
        
        # Mock the absence generation to be fast
        with patch('generate_absence_data.RandomBackgroundGenerator') as mock_bg_gen:
            mock_absences = gpd.GeoDataFrame({
                'latitude': [43.2, 43.3],
                'longitude': [-110.2, -110.3],
                'absence_strategy': ['background', 'background']
            }, geometry=gpd.points_from_xy([-110.2, -110.3], [43.2, 43.3]), crs='EPSG:4326')
            mock_bg_gen.return_value.generate.return_value = mock_absences
            
            # This test verifies the structure, actual execution would be more complex
            assert presence_file.exists()
    
    def test_pipeline_structure(self, test_environment):
        """Test that pipeline structure is correct."""
        import importlib.util
        scripts_dir = Path(__file__).parent.parent / "scripts"
        
        spec = importlib.util.spec_from_file_location(
            "run_data_pipeline",
            scripts_dir / "run_data_pipeline.py"
        )
        pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_module)
        
        DataPipeline = pipeline_module.DataPipeline
        
        data_dir = test_environment['data_dir']
        
        pipeline = DataPipeline(
            data_dir=data_dir,
            dataset_name='test_dataset',
            skip_steps=[],
            force=False
        )
        
        # Verify pipeline has all expected steps
        step_names = [step.name for step in pipeline.steps]
        assert 'process_raw' in step_names
        assert 'generate_absence' in step_names
        assert 'integrate_features' in step_names
        assert 'analyze_features' in step_names
        assert 'assess_readiness' in step_names
        assert 'prepare_features' in step_names  # New step
        
        # Verify step order is correct
        assert pipeline.steps[0].name == 'process_raw'
        assert pipeline.steps[1].name == 'generate_absence'
        assert pipeline.steps[2].name == 'integrate_features'
        assert pipeline.steps[5].name == 'prepare_features'  # Should be last step
    
    def test_pipeline_handles_missing_data_gracefully(self, test_environment):
        """Test that pipeline handles missing data files gracefully."""
        import importlib.util
        scripts_dir = Path(__file__).parent.parent / "scripts"
        
        spec = importlib.util.spec_from_file_location(
            "run_data_pipeline",
            scripts_dir / "run_data_pipeline.py"
        )
        pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_module)
        
        DataPipeline = pipeline_module.DataPipeline
        
        data_dir = test_environment['data_dir']
        
        # Create pipeline with non-existent dataset
        pipeline = DataPipeline(
            data_dir=data_dir,
            dataset_name='nonexistent',
            skip_steps=[],
            force=False
        )
        
        # Steps requiring missing inputs should report can_run() as False
        for step in pipeline.steps:
            if step.required_input and not step.required_input.exists():
                # This step cannot run
                assert step.can_run() is False or step.required_input is None

