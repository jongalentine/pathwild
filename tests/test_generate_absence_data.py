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

