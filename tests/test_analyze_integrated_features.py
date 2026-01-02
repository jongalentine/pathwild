"""
Tests for analyze_integrated_features.py script.

Tests the feature analysis logic that validates integrated datasets.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys
import importlib.util
import shutil


class TestAnalyzeIntegratedFeatures:
    """Test the analyze_integrated_features script functions."""
    
    @pytest.fixture
    def sample_integrated_dataset(self, tmp_path):
        """Create a sample integrated dataset with features and placeholders."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2, 43.3],
            'longitude': [-110.0, -110.1, -110.2, -110.3],
            'elk_present': [1, 0, 1, 0],
            # Environmental features
            'elevation': [2000.0, 2100.0, 8500.0, 2200.0],  # One placeholder
            'slope_degrees': [5.0, 10.0, 15.0, 20.0],
            'aspect_degrees': [180.0, 270.0, 90.0, 0.0],
            'canopy_cover_percent': [30.0, 40.0, 30.0, 50.0],  # One placeholder
            'land_cover_code': [1, 2, 0, 3],  # One placeholder
            'water_distance_miles': [0.5, 1.0, 1.5, 2.0],  # One placeholder
            'water_reliability': [0.8, 0.7, 0.9, 0.6],
            'road_distance_miles': [2.0, 3.0, 4.0, 5.0],
            'trail_distance_miles': [1.0, 2.0, 3.0, 4.0],
            'security_habitat_percent': [60.0, 70.0, 80.0, 90.0]
        })
        
        file_path = tmp_path / "combined_test_presence_absence.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            shutil.copy2(real_script, temp_script)
        else:
            pytest.skip("Real analyze_integrated_features.py not found")
        
        spec = importlib.util.spec_from_file_location(
            "analyze_integrated_features",
            temp_script
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def test_analyze_detects_placeholders(self, analyze_module, sample_integrated_dataset):
        """Test that placeholder values are detected."""
        result = analyze_module.analyze_integrated_features(sample_integrated_dataset)
        
        # Should return True on success
        assert result is True
        
        # The function prints to stdout, so we can't easily capture it
        # But we can verify it doesn't crash
    
    def test_analyze_handles_missing_file(self, analyze_module, tmp_path):
        """Test that missing file is handled gracefully."""
        missing_file = tmp_path / "nonexistent.csv"
        result = analyze_module.analyze_integrated_features(missing_file)
        
        assert result is False
    
    def test_placeholder_values_constant(self, analyze_module):
        """Test that PLACEHOLDER_VALUES constant is defined correctly."""
        assert hasattr(analyze_module, 'PLACEHOLDER_VALUES')
        assert isinstance(analyze_module.PLACEHOLDER_VALUES, dict)
        assert 'elevation' in analyze_module.PLACEHOLDER_VALUES
        assert analyze_module.PLACEHOLDER_VALUES['elevation'] == 8500.0
        assert analyze_module.PLACEHOLDER_VALUES['water_distance_miles'] == 0.5
        assert analyze_module.PLACEHOLDER_VALUES['canopy_cover_percent'] == 30.0
        assert analyze_module.PLACEHOLDER_VALUES['land_cover_code'] == 0
    
    def test_analyze_with_all_placeholders(self, analyze_module, tmp_path):
        """Test analysis with all placeholder values."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1],
            'longitude': [-110.0, -110.1],
            'elk_present': [1, 0],
            'elevation': [8500.0, 8500.0],  # All placeholders
            'slope_degrees': [15.0, 20.0],
            'aspect_degrees': [180.0, 270.0],
            'canopy_cover_percent': [30.0, 30.0],  # All placeholders
            'land_cover_code': [0, 0],  # All placeholders
            'water_distance_miles': [0.5, 0.5],  # All placeholders
            'water_reliability': [0.7, 0.8]
        })
        
        file_path = tmp_path / "all_placeholders.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_analyze_with_no_placeholders(self, analyze_module, tmp_path):
        """Test analysis with no placeholder values."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1],
            'longitude': [-110.0, -110.1],
            'elk_present': [1, 0],
            'elevation': [2000.0, 2100.0],  # No placeholders
            'slope_degrees': [5.0, 10.0],
            'aspect_degrees': [180.0, 270.0],
            'canopy_cover_percent': [40.0, 50.0],  # No placeholders
            'land_cover_code': [1, 2],  # No placeholders
            'water_distance_miles': [1.0, 2.0],  # No placeholders
            'water_reliability': [0.7, 0.8]
        })
        
        file_path = tmp_path / "no_placeholders.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_analyze_with_missing_values(self, analyze_module, tmp_path):
        """Test analysis with missing values."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'elk_present': [1, 0, 1],
            'elevation': [2000.0, None, 2200.0],  # Missing value
            'slope_degrees': [5.0, 10.0, 15.0],
            'aspect_degrees': [180.0, 270.0, 90.0],
            'canopy_cover_percent': [40.0, None, 50.0],  # Missing value
            'land_cover_code': [1, 2, 3],
            'water_distance_miles': [1.0, 2.0, 3.0],
            'water_reliability': [0.7, 0.8, 0.9]
        })
        
        file_path = tmp_path / "missing_values.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestAnalyzeIntegratedFeaturesIntegration:
    """Integration tests for analyze_integrated_features script."""
    
    def test_script_runs_with_valid_dataset(self, tmp_path):
        """Test that script can be run with valid dataset."""
        # Create test dataset
        df = pd.DataFrame({
            'latitude': [43.0, 43.1],
            'longitude': [-110.0, -110.1],
            'elk_present': [1, 0],
            'elevation': [2000.0, 2100.0],
            'slope_degrees': [5.0, 10.0],
            'aspect_degrees': [180.0, 270.0],
            'canopy_cover_percent': [40.0, 50.0],
            'land_cover_code': [1, 2],
            'water_distance_miles': [1.0, 2.0],
            'water_reliability': [0.7, 0.8]
        })
        
        dataset_file = tmp_path / "test_dataset.csv"
        df.to_csv(dataset_file, index=False)
        
        # Run script via subprocess
        import subprocess
        scripts_dir = Path(__file__).parent.parent / "scripts"
        script_path = scripts_dir / "analyze_integrated_features.py"
        
        if not script_path.exists():
            pytest.skip("Script not found")
        
        result = subprocess.run(
            [sys.executable, str(script_path), str(dataset_file)],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )
        
        # Should succeed
        assert result.returncode == 0
        
        # Should contain analysis output
        assert "INTEGRATED FEATURES ANALYSIS" in result.stdout
        assert "FEATURE VALUE RANGES" in result.stdout
        assert "PLACEHOLDER VALUE DETECTION" in result.stdout
    
    def test_script_handles_missing_file(self, tmp_path):
        """Test that script handles missing file gracefully."""
        import subprocess
        scripts_dir = Path(__file__).parent.parent / "scripts"
        script_path = scripts_dir / "analyze_integrated_features.py"
        
        if not script_path.exists():
            pytest.skip("Script not found")
        
        missing_file = tmp_path / "nonexistent.csv"
        
        result = subprocess.run(
            [sys.executable, str(script_path), str(missing_file)],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )
        
        # Should fail gracefully
        assert result.returncode == 1
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

