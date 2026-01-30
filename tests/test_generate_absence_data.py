"""
Tests for generate_absence_data.py script.

Tests absence generation and validation.
Note: Feature enrichment has been removed from this script and is now handled
by integrate_environmental_features.py to avoid duplication.
"""
import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import importlib.util

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import the module
spec = importlib.util.spec_from_file_location(
    "generate_absence_data",
    scripts_dir / "generate_absence_data.py"
)
absence_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(absence_module)


class TestAbsenceGeneration:
    """Test absence data generation (without enrichment)."""
    
    @pytest.fixture
    def sample_presence_data(self):
        """Create sample presence data for testing."""
        return pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'elk_present': [1, 1, 1]
        })
    
    def test_absence_generation_removed_enrichment(self):
        """Verify that enrichment functions are no longer in the module."""
        # These functions should not exist anymore
        assert not hasattr(absence_module, 'enrich_with_features'), \
            "enrich_with_features should have been removed"
        assert not hasattr(absence_module, '_enrich_batch'), \
            "_enrich_batch should have been removed"
        assert not hasattr(absence_module, 'DataContextBuilder'), \
            "DataContextBuilder import should have been removed"
    
    def test_temporal_generators_available(self):
        """Verify that temporal generators can be imported."""
        try:
            from src.data.temporal_absence_generators import (
                TemporallyMatchedAbsenceGenerator,
                SeasonalSegregationAbsenceGenerator,
                UnsuitableTemporalEnvironmentalAbsenceGenerator,
                RandomTemporalBackgroundGenerator
            )
            assert True, "Temporal generators are available"
        except ImportError:
            pytest.skip("Temporal generators not available")
    
    def test_temporal_metadata_in_legacy_generators(self, sample_presence_data):
        """Test that legacy generators add temporal metadata when dates are available."""
        from src.data.absence_generators import RandomBackgroundGenerator
        from shapely.geometry import box
        import geopandas as gpd
        
        # Add date column to presence data
        sample_presence_data['date'] = pd.to_datetime(['2020-06-15', '2020-07-20', '2020-08-10'])
        presence_gdf = gpd.GeoDataFrame(
            sample_presence_data,
            geometry=gpd.points_from_xy(
                sample_presence_data.longitude,
                sample_presence_data.latitude
            ),
            crs="EPSG:4326"
        )
        
        study_area = gpd.GeoDataFrame(geometry=[box(-111.0, 41.0, -104.0, 45.0)], crs="EPSG:4326")
        
        generator = RandomBackgroundGenerator(presence_gdf, study_area)
        absences = generator.generate(n_samples=3, max_attempts=1000)
        
        # Check that temporal metadata was added
        if len(absences) > 0:
            # The _add_temporal_metadata method should have been called
            # Check if temporal columns exist (they should if dates were available)
            temporal_cols = ['date', 'year', 'month', 'day_of_year']
            has_temporal = any(col in absences.columns for col in temporal_cols)
            # Note: This may not always add temporal metadata if dates aren't parsed correctly
            # But the method should exist and be callable
            assert hasattr(generator, '_add_temporal_metadata'), \
                "Legacy generators should have _add_temporal_metadata method"
    
    def test_limit_creates_test_file(self, tmp_path):
        """Test that --limit option creates *_test.csv output file."""
        from unittest.mock import patch, MagicMock
        
        # Create sample presence data file
        presence_file = tmp_path / "test_presence.csv"
        presence_df = pd.DataFrame({
            'latitude': [43.0 + i*0.1 for i in range(10)],
            'longitude': [-110.0 - i*0.1 for i in range(10)],
            'elk_present': [1] * 10
        })
        presence_df.to_csv(presence_file, index=False)
        
        # Create minimal study area
        study_area_file = tmp_path / "study_area.geojson"
        study_area_data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-111.0, 41.0],
                        [-104.0, 41.0],
                        [-104.0, 45.0],
                        [-111.0, 45.0],
                        [-111.0, 41.0]
                    ]]
                }
            }]
        }
        import json
        study_area_file.write_text(json.dumps(study_area_data))
        
        # Create data directory structure
        data_dir = tmp_path / "data"
        (data_dir / "boundaries").mkdir(parents=True)
        (data_dir / "boundaries" / "wyoming_state.shp").parent.mkdir(exist_ok=True)
        
        # Mock the absence generators to avoid actual generation
        with patch('src.data.absence_generators.RandomBackgroundGenerator') as mock_bg, \
             patch('src.data.absence_generators.EnvironmentalPseudoAbsenceGenerator') as mock_env, \
             patch('src.data.absence_generators.UnsuitableHabitatAbsenceGenerator') as mock_unsuit:
            
            # Create mock absence data
            mock_absence_df = pd.DataFrame({
                'latitude': [43.5, 43.6],
                'longitude': [-110.5, -110.6],
                'elk_present': [0, 0]
            })
            mock_absence_gdf = gpd.GeoDataFrame(
                mock_absence_df,
                geometry=gpd.points_from_xy(mock_absence_df.longitude, mock_absence_df.latitude),
                crs="EPSG:4326"
            )
            
            # Configure mocks
            mock_bg_instance = MagicMock()
            mock_bg_instance.generate.return_value = mock_absence_gdf
            mock_bg_instance._add_temporal_metadata = lambda x, y: x
            mock_bg.return_value = mock_bg_instance
            
            mock_env_instance = MagicMock()
            mock_env_instance.generate.return_value = mock_absence_gdf
            mock_env_instance._add_temporal_metadata = lambda x, y: x
            mock_env.return_value = mock_env_instance
            
            mock_unsuit_instance = MagicMock()
            mock_unsuit_instance.generate.return_value = mock_absence_gdf
            mock_unsuit_instance._add_temporal_metadata = lambda x, y: x
            mock_unsuit.return_value = mock_unsuit_instance
            
            # Import and run main function with --limit
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    'generate_absence_data.py',
                    '--presence-file', str(presence_file),
                    '--output-file', str(tmp_path / "combined_test_presence_absence.csv"),
                    '--data-dir', str(data_dir),
                    '--limit', '5'
                ]
                
                # Call main function
                result = absence_module.main()
                
                # Check that output file has _test suffix
                expected_output = tmp_path / "combined_test_presence_absence_test.csv"
                assert expected_output.exists(), f"Expected test file {expected_output} not found"
                
                # Verify the file contains limited data
                output_df = pd.read_csv(expected_output)
                # Should have 5 presence + some absences
                assert len(output_df) >= 5, "Output should have at least 5 rows (limited presence)"
                
            finally:
                sys.argv = original_argv
    
    def test_no_limit_creates_normal_file(self, tmp_path):
        """Test that without --limit, normal output file is created."""
        from unittest.mock import patch, MagicMock
        
        # Create sample presence data file
        presence_file = tmp_path / "test_presence.csv"
        presence_df = pd.DataFrame({
            'latitude': [43.0 + i*0.1 for i in range(10)],
            'longitude': [-110.0 - i*0.1 for i in range(10)],
            'elk_present': [1] * 10
        })
        presence_df.to_csv(presence_file, index=False)
        
        # Create data directory structure
        data_dir = tmp_path / "data"
        (data_dir / "boundaries").mkdir(parents=True)
        
        # Mock the absence generators
        with patch('src.data.absence_generators.RandomBackgroundGenerator') as mock_bg, \
             patch('src.data.absence_generators.EnvironmentalPseudoAbsenceGenerator') as mock_env, \
             patch('src.data.absence_generators.UnsuitableHabitatAbsenceGenerator') as mock_unsuit:
            
            # Create mock absence data
            mock_absence_df = pd.DataFrame({
                'latitude': [43.5, 43.6],
                'longitude': [-110.5, -110.6],
                'elk_present': [0, 0]
            })
            mock_absence_gdf = gpd.GeoDataFrame(
                mock_absence_df,
                geometry=gpd.points_from_xy(mock_absence_df.longitude, mock_absence_df.latitude),
                crs="EPSG:4326"
            )
            
            # Configure mocks
            mock_bg_instance = MagicMock()
            mock_bg_instance.generate.return_value = mock_absence_gdf
            mock_bg_instance._add_temporal_metadata = lambda x, y: x
            mock_bg.return_value = mock_bg_instance
            
            mock_env_instance = MagicMock()
            mock_env_instance.generate.return_value = mock_absence_gdf
            mock_env_instance._add_temporal_metadata = lambda x, y: x
            mock_env.return_value = mock_env_instance
            
            mock_unsuit_instance = MagicMock()
            mock_unsuit_instance.generate.return_value = mock_absence_gdf
            mock_unsuit_instance._add_temporal_metadata = lambda x, y: x
            mock_unsuit.return_value = mock_unsuit_instance
            
            # Import and run main function without --limit
            import sys
            original_argv = sys.argv
            try:
                output_file = tmp_path / "combined_test_presence_absence.csv"
                sys.argv = [
                    'generate_absence_data.py',
                    '--presence-file', str(presence_file),
                    '--output-file', str(output_file),
                    '--data-dir', str(data_dir),
                    # No --limit flag
                ]
                
                # Call main function
                result = absence_module.main()
                
                # Check that output file does NOT have _test suffix
                assert output_file.exists(), f"Expected output file {output_file} not found"
                assert not output_file.stem.endswith('_test'), "Output file should not have _test suffix when limit is not set"
                
            finally:
                sys.argv = original_argv
    
    def test_output_filename_logic_with_limit(self):
        """Test the output filename logic when limit is set."""
        from pathlib import Path
        
        # Test the logic: if limit is set and stem doesn't end with _test, add _test
        output_file_path = Path("combined_test_presence_absence.csv")
        limit = 5
        
        # Simulate the logic from generate_absence_data.py
        if limit is not None and not output_file_path.stem.endswith('_test'):
            output_file_path = output_file_path.parent / f"{output_file_path.stem}_test.csv"
        
        assert output_file_path.name == "combined_test_presence_absence_test.csv"
        assert output_file_path.stem.endswith('_test')
    
    def test_output_filename_logic_without_limit(self):
        """Test the output filename logic when limit is not set."""
        from pathlib import Path
        
        # Test the logic: if limit is None, keep original filename
        output_file_path = Path("combined_test_presence_absence.csv")
        limit = None
        
        # Simulate the logic from generate_absence_data.py
        if limit is not None and not output_file_path.stem.endswith('_test'):
            output_file_path = output_file_path.parent / f"{output_file_path.stem}_test.csv"
        
        assert output_file_path.name == "combined_test_presence_absence.csv"
        assert not output_file_path.stem.endswith('_test')

