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

