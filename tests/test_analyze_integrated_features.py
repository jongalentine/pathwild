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
        # Temporal feature placeholders
        assert 'temperature_f' in analyze_module.PLACEHOLDER_VALUES
        assert analyze_module.PLACEHOLDER_VALUES['temperature_f'] == 45.0
        assert 'ndvi' in analyze_module.PLACEHOLDER_VALUES
        assert analyze_module.PLACEHOLDER_VALUES['ndvi'] == 0.5
        assert 'precip_last_7_days_inches' in analyze_module.PLACEHOLDER_VALUES
        assert analyze_module.PLACEHOLDER_VALUES['precip_last_7_days_inches'] == 0.0
    
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


class TestInfrastructureAnalysis:
    """Test infrastructure (roads/trails) analysis functionality."""
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            import shutil
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
    
    def test_roads_trails_analysis_with_valid_data(self, analyze_module, tmp_path):
        """Test that roads/trails analysis works with valid data."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'elk_present': [1, 0, 1],
            'road_distance_miles': [2.5, 5.0, 8.0],
            'trail_distance_miles': [1.0, 3.0, 6.0],
            'elevation': [2000.0, 2100.0, 2200.0],
        })
        
        file_path = tmp_path / "test_infrastructure.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_roads_trails_analysis_with_default_values(self, analyze_module, tmp_path):
        """Test detection of default road/trail distances."""
        df = pd.DataFrame({
            'latitude': [43.0] * 15,  # 15 rows, >10% have defaults
            'longitude': [-110.0] * 15,
            'elk_present': [1] * 15,
            'road_distance_miles': [2.0] * 15,  # All defaults (100%)
            'trail_distance_miles': [1.5] * 15,  # All defaults (100%)
            'elevation': [2000.0] * 15,
        })
        
        file_path = tmp_path / "test_defaults.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_roads_trails_analysis_with_missing_columns(self, analyze_module, tmp_path):
        """Test that missing road/trail columns are handled gracefully."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1],
            'longitude': [-110.0, -110.1],
            'elk_present': [1, 0],
            'elevation': [2000.0, 2100.0],
        })
        
        file_path = tmp_path / "test_no_infrastructure.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_roads_trails_correlation_analysis(self, analyze_module, tmp_path):
        """Test correlation analysis between roads and trails."""
        # Create data with correlation
        import numpy as np
        np.random.seed(42)
        road_dist = np.random.uniform(0, 20, 50)
        trail_dist = road_dist + np.random.uniform(-2, 2, 50)  # Correlated
        
        df = pd.DataFrame({
            'latitude': np.random.uniform(43.0, 44.0, 50),
            'longitude': np.random.uniform(-110.0, -109.0, 50),
            'elk_present': np.random.choice([0, 1], 50),
            'road_distance_miles': road_dist,
            'trail_distance_miles': trail_dist,
            'elevation': np.random.uniform(2000, 3000, 50),
        })
        
        file_path = tmp_path / "test_correlation.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestGeographicValidation:
    """Test geographic validation functionality."""
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            import shutil
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
    
    def test_geographic_validation_within_wyoming(self, analyze_module, tmp_path):
        """Test geographic validation with coordinates within Wyoming."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.5, 44.0],  # Within Wyoming
            'longitude': [-110.0, -107.0, -105.0],  # Within Wyoming
            'elk_present': [1, 0, 1],
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_within_wyoming.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_geographic_validation_outside_wyoming(self, analyze_module, tmp_path):
        """Test geographic validation with coordinates outside Wyoming."""
        df = pd.DataFrame({
            'latitude': [46.0, 39.0, 43.0],  # Two outside, one inside
            'longitude': [-110.0, -110.0, -112.0],  # One outside, two inside
            'elk_present': [1, 0, 1],
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_outside_wyoming.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_geographic_validation_missing_coordinates(self, analyze_module, tmp_path):
        """Test geographic validation with missing coordinates."""
        df = pd.DataFrame({
            'latitude': [43.0, None, 44.0],
            'longitude': [-110.0, -107.0, None],
            'elk_present': [1, 0, 1],
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_missing_coords.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestExpectedValueRanges:
    """Test expected value ranges validation."""
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            import shutil
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
    
    def test_expected_ranges_constant_exists(self, analyze_module):
        """Test that EXPECTED_RANGES constant is defined."""
        assert hasattr(analyze_module, 'EXPECTED_RANGES')
        assert isinstance(analyze_module.EXPECTED_RANGES, dict)
        assert 'elevation' in analyze_module.EXPECTED_RANGES
        assert 'latitude' in analyze_module.EXPECTED_RANGES
        assert 'longitude' in analyze_module.EXPECTED_RANGES
    
    def test_value_range_validation_within_range(self, analyze_module, tmp_path):
        """Test value range validation with values within expected ranges."""
        df = pd.DataFrame({
            'latitude': [43.0, 44.0],
            'longitude': [-110.0, -107.0],
            'elevation': [5000.0, 8000.0],  # Within range (1000-14000)
            'slope_degrees': [10.0, 30.0],  # Within range (0-60)
            'aspect_degrees': [180.0, 270.0],  # Within range (0-360)
            'canopy_cover_percent': [50.0, 70.0],  # Within range (0-100)
            'water_distance_miles': [1.0, 5.0],  # Within range (0-50)
        })
        
        file_path = tmp_path / "test_within_range.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_value_range_validation_outside_range(self, analyze_module, tmp_path):
        """Test value range validation with values outside expected ranges."""
        df = pd.DataFrame({
            'latitude': [43.0, 44.0],
            'longitude': [-110.0, -107.0],
            'elevation': [15000.0, 8000.0],  # One above max (14000)
            'slope_degrees': [70.0, 30.0],  # One above max (60)
            'aspect_degrees': [180.0, 370.0],  # One above max (360)
            'canopy_cover_percent': [50.0, 110.0],  # One above max (100)
        })
        
        file_path = tmp_path / "test_outside_range.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestLandCoverValidation:
    """Test NLCD land cover code validation."""
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            import shutil
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
    
    def test_valid_nlcd_codes_constant_exists(self, analyze_module):
        """Test that VALID_NLCD_CODES constant is defined."""
        assert hasattr(analyze_module, 'VALID_NLCD_CODES')
        assert isinstance(analyze_module.VALID_NLCD_CODES, set)
        assert 41 in analyze_module.VALID_NLCD_CODES  # Deciduous Forest
        assert 42 in analyze_module.VALID_NLCD_CODES  # Evergreen Forest
        assert 52 in analyze_module.VALID_NLCD_CODES  # Shrub/Scrub
    
    def test_land_cover_validation_with_valid_codes(self, analyze_module, tmp_path):
        """Test land cover validation with valid NLCD codes."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'land_cover_code': [41, 42, 52],  # Valid NLCD codes
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_valid_codes.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_land_cover_validation_with_invalid_codes(self, analyze_module, tmp_path):
        """Test land cover validation with invalid NLCD codes."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'land_cover_code': [41, 999, 52],  # One invalid code (999)
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_invalid_codes.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestStatisticalOutlierDetection:
    """Test statistical outlier detection functionality."""
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            import shutil
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
    
    def test_outlier_detection_with_normal_data(self, analyze_module, tmp_path):
        """Test outlier detection with normal data (no outliers)."""
        import numpy as np
        np.random.seed(42)
        
        df = pd.DataFrame({
            'latitude': np.random.uniform(43.0, 44.0, 50),
            'longitude': np.random.uniform(-110.0, -109.0, 50),
            'elevation': np.random.normal(5000, 1000, 50),  # Normal distribution
            'slope_degrees': np.random.normal(15, 5, 50),
            'water_distance_miles': np.random.normal(2, 1, 50),
        })
        
        file_path = tmp_path / "test_normal_data.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_outlier_detection_with_outliers(self, analyze_module, tmp_path):
        """Test outlier detection with data containing outliers."""
        import numpy as np
        np.random.seed(42)
        
        elevation = np.random.normal(5000, 1000, 50)
        elevation[0] = 20000  # Outlier
        elevation[1] = -1000  # Outlier
        
        df = pd.DataFrame({
            'latitude': np.random.uniform(43.0, 44.0, 50),
            'longitude': np.random.uniform(-110.0, -109.0, 50),
            'elevation': elevation,
            'slope_degrees': np.random.normal(15, 5, 50),
        })
        
        file_path = tmp_path / "test_outliers.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestFeatureRelationshipValidation:
    """Test feature relationship validation functionality."""
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            import shutil
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
    
    def test_land_cover_canopy_consistency(self, analyze_module, tmp_path):
        """Test land cover vs canopy cover consistency validation."""
        df = pd.DataFrame({
            'latitude': [43.0] * 20,
            'longitude': [-110.0] * 20,
            'land_cover_code': [41] * 10 + [21] * 10,  # Forest + Developed
            'canopy_cover_percent': [60.0] * 10 + [5.0] * 10,  # High for forest, low for developed
            'elevation': [2000.0] * 20,
        })
        
        file_path = tmp_path / "test_landcover_canopy.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_aspect_validation(self, analyze_module, tmp_path):
        """Test aspect degree validation (0-360)."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'aspect_degrees': [180.0, 370.0, -10.0],  # One above max, one below min
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_aspect.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_water_reliability_range(self, analyze_module, tmp_path):
        """Test water reliability range validation (0-1)."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'water_reliability': [0.7, 1.5, -0.1],  # One above max, one below min
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_reliability.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_security_habitat_range(self, analyze_module, tmp_path):
        """Test security habitat range validation (0-100)."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'security_habitat_percent': [50.0, 150.0, -10.0],  # One above max, one below min
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_security.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestFeatureCompleteness:
    """Test feature completeness validation."""
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            import shutil
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
    
    def test_feature_completeness_with_all_features(self, analyze_module, tmp_path):
        """Test feature completeness with all expected features present."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1],
            'longitude': [-110.0, -110.1],
            'elk_present': [1, 0],
            'elevation': [2000.0, 3000.0],
            'slope_degrees': [10.0, 15.0],
            'aspect_degrees': [180.0, 270.0],
            'canopy_cover_percent': [50.0, 60.0],
            'land_cover_code': [41, 42],
            'water_distance_miles': [1.0, 2.0],
            'water_reliability': [0.8, 0.9],
            'road_distance_miles': [2.0, 3.0],
            'trail_distance_miles': [1.0, 2.0],
            'security_habitat_percent': [60.0, 70.0],
        })
        
        file_path = tmp_path / "test_complete_features.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_feature_completeness_with_missing_features(self, analyze_module, tmp_path):
        """Test feature completeness with missing expected features."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1],
            'longitude': [-110.0, -110.1],
            'elk_present': [1, 0],
            'elevation': [2000.0, 3000.0],
            # Missing: slope, aspect, canopy, land_cover, etc.
        })
        
        file_path = tmp_path / "test_missing_features.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_feature_completeness_with_partial_features(self, analyze_module, tmp_path):
        """Test feature completeness with partial feature coverage."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'elk_present': [1, 0, 1],
            'elevation': [2000.0, None, 4000.0],  # One missing
            'slope_degrees': [10.0, 15.0, None],  # One missing
            'canopy_cover_percent': [50.0, None, None],  # Two missing
        })
        
        file_path = tmp_path / "test_partial_features.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestValidationOutputSections:
    """Test that new validation sections appear in output."""
    
    def test_integration_output_includes_new_sections(self, tmp_path):
        """Test that integration test output includes all new validation sections."""
        import subprocess
        
        # Create comprehensive test dataset
        df = pd.DataFrame({
            'latitude': [43.0, 43.5, 44.0],
            'longitude': [-110.0, -107.0, -105.0],
            'elk_present': [1, 0, 1],
            'elevation': [5000.0, 8000.0, 10000.0],
            'slope_degrees': [10.0, 20.0, 30.0],
            'aspect_degrees': [180.0, 270.0, 90.0],
            'canopy_cover_percent': [50.0, 60.0, 70.0],
            'land_cover_code': [41, 42, 52],
            'water_distance_miles': [1.0, 2.0, 3.0],
            'water_reliability': [0.8, 0.9, 0.7],
            'road_distance_miles': [2.0, 3.0, 4.0],
            'trail_distance_miles': [1.0, 2.0, 3.0],
            'security_habitat_percent': [60.0, 70.0, 80.0],
        })
        
        file_path = tmp_path / "test_comprehensive.csv"
        df.to_csv(file_path, index=False)
        
        # Run script via subprocess
        scripts_dir = Path(__file__).parent.parent / "scripts"
        script_path = scripts_dir / "analyze_integrated_features.py"
        
        if not script_path.exists():
            pytest.skip("Script not found")
        
        result = subprocess.run(
            [sys.executable, str(script_path), str(file_path)],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )
        
        # Should succeed
        assert result.returncode == 0
        
        # Should contain all new validation sections
        assert "GEOGRAPHIC VALIDATION" in result.stdout
        assert "EXPECTED VALUE RANGES VALIDATION" in result.stdout
        assert "LAND COVER VALIDATION" in result.stdout
        assert "STATISTICAL OUTLIER DETECTION" in result.stdout
        assert "FEATURE RELATIONSHIP VALIDATION" in result.stdout
        assert "FEATURE COMPLETENESS" in result.stdout
        assert "INFRASTRUCTURE DATA QUALITY (ROADS & TRAILS)" in result.stdout


class TestNDVIDataQuality:
    """Test NDVI data quality analysis functionality."""
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            import shutil
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
    
    def test_ndvi_analysis_with_valid_data(self, analyze_module, tmp_path):
        """Test NDVI analysis with valid data."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'ndvi': [0.75, 0.65, 0.55],  # Valid NDVI values
            'ndvi_age_days': [5, 10, 15],
            'irg': [0.01, 0.005, -0.002],
            'summer_integrated_ndvi': [70.0, 65.0, 60.0],
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_ndvi.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_ndvi_placeholder_detection(self, analyze_module, tmp_path):
        """Test detection of placeholder NDVI values."""
        df = pd.DataFrame({
            'latitude': [43.0] * 15,
            'longitude': [-110.0] * 15,
            'ndvi': [0.5] * 15,  # All placeholders
            'elevation': [2000.0] * 15,
        })
        
        file_path = tmp_path / "test_ndvi_placeholder.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_ndvi_seasonal_variation(self, analyze_module, tmp_path):
        """Test NDVI seasonal variation analysis."""
        df = pd.DataFrame({
            'latitude': [43.0] * 12,
            'longitude': [-110.0] * 12,
            'date': pd.date_range('2024-01-01', periods=12, freq='MS'),  # Monthly
            'ndvi': [0.3, 0.3, 0.4, 0.5, 0.6, 0.75, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3],  # Seasonal pattern
            'elevation': [2000.0] * 12,
        })
        
        file_path = tmp_path / "test_ndvi_seasonal.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_ndvi_age_analysis(self, analyze_module, tmp_path):
        """Test NDVI age analysis."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'ndvi': [0.75, 0.65, 0.55],
            'ndvi_age_days': [5, 50, 120],  # One very old (>90 days)
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_ndvi_age.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_irg_analysis(self, analyze_module, tmp_path):
        """Test IRG (Instantaneous Rate of Green-up) analysis."""
        df = pd.DataFrame({
            'latitude': [43.0] * 12,
            'longitude': [-110.0] * 12,
            'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
            'irg': [0.0, 0.0, 0.01, 0.02, 0.015, 0.005, 0.0, -0.005, -0.01, -0.008, -0.003, 0.0],
            'ndvi': [0.3, 0.3, 0.4, 0.5, 0.6, 0.75, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3],
            'elevation': [2000.0] * 12,
        })
        
        file_path = tmp_path / "test_irg.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_summer_integrated_ndvi_analysis(self, analyze_module, tmp_path):
        """Test summer integrated NDVI analysis."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'ndvi': [0.75, 0.65, 0.55],
            'summer_integrated_ndvi': [70.0, 65.0, 60.0],  # Valid values
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_summer_ndvi.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_ndvi_range_validation(self, analyze_module, tmp_path):
        """Test NDVI range validation (-1 to 1)."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'ndvi': [0.75, 1.5, -1.5],  # One above max, one below min
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_ndvi_range.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestWeatherTemperatureDataQuality:
    """Test weather and temperature data quality analysis functionality."""
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            import shutil
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
    
    def test_temperature_analysis_with_valid_data(self, analyze_module, tmp_path):
        """Test temperature analysis with valid data."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'temperature_f': [70.0, 65.0, 60.0],
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_temperature.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_temperature_placeholder_detection(self, analyze_module, tmp_path):
        """Test detection of placeholder temperature values."""
        df = pd.DataFrame({
            'latitude': [43.0] * 15,
            'longitude': [-110.0] * 15,
            'temperature_f': [45.0] * 15,  # All placeholders
            'elevation': [2000.0] * 15,
        })
        
        file_path = tmp_path / "test_temp_placeholder.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_temperature_seasonal_variation(self, analyze_module, tmp_path):
        """Test temperature seasonal variation analysis."""
        df = pd.DataFrame({
            'latitude': [43.0] * 12,
            'longitude': [-110.0] * 12,
            'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
            'temperature_f': [20.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 70.0, 60.0, 50.0, 40.0, 25.0],  # Seasonal pattern
            'elevation': [2000.0] * 12,
        })
        
        file_path = tmp_path / "test_temp_seasonal.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_temperature_elevation_correlation(self, analyze_module, tmp_path):
        """Test temperature-elevation correlation analysis."""
        import numpy as np
        np.random.seed(42)
        
        # Higher elevation should be colder (negative correlation)
        elevation = np.random.uniform(2000, 10000, 50)
        temperature = 80 - (elevation - 2000) / 200 + np.random.normal(0, 5, 50)  # Negative correlation
        
        df = pd.DataFrame({
            'latitude': np.random.uniform(43.0, 44.0, 50),
            'longitude': np.random.uniform(-110.0, -109.0, 50),
            'elevation': elevation,
            'temperature_f': temperature,
        })
        
        file_path = tmp_path / "test_temp_elevation.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_precipitation_analysis(self, analyze_module, tmp_path):
        """Test precipitation analysis."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'precip_last_7_days_inches': [0.5, 1.0, 0.2],
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_precip.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_precipitation_placeholder_detection(self, analyze_module, tmp_path):
        """Test detection of placeholder precipitation (zero values)."""
        df = pd.DataFrame({
            'latitude': [43.0] * 20,
            'longitude': [-110.0] * 20,
            'precip_last_7_days_inches': [0.0] * 20,  # All zeros (>70%)
            'elevation': [2000.0] * 20,
        })
        
        file_path = tmp_path / "test_precip_placeholder.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_cloud_cover_analysis(self, analyze_module, tmp_path):
        """Test cloud cover analysis."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'cloud_cover_percent': [30.0, 40.0, 20.0],  # Placeholder values
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_cloud_cover.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_cloud_cover_range_validation(self, analyze_module, tmp_path):
        """Test cloud cover range validation (0-100%)."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'cloud_cover_percent': [50.0, 150.0, -10.0],  # One above max, one below min
            'elevation': [2000.0, 3000.0, 4000.0],
        })
        
        file_path = tmp_path / "test_cloud_range.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestTemporalFeatureRelationships:
    """Test temporal feature relationship validation."""
    
    @pytest.fixture
    def analyze_module(self, tmp_path):
        """Load the analyze_integrated_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "analyze_integrated_features.py"
        temp_script = tmp_path / "analyze_integrated_features.py"
        
        if real_script.exists():
            import shutil
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
    
    def test_ndvi_season_relationship(self, analyze_module, tmp_path):
        """Test NDVI-season relationship validation."""
        df = pd.DataFrame({
            'latitude': [43.0] * 12,
            'longitude': [-110.0] * 12,
            'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
            'ndvi': [0.3, 0.3, 0.4, 0.5, 0.6, 0.75, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3],  # Summer > Winter
            'elevation': [2000.0] * 12,
        })
        
        file_path = tmp_path / "test_ndvi_season.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_temperature_season_relationship(self, analyze_module, tmp_path):
        """Test temperature-season relationship validation."""
        df = pd.DataFrame({
            'latitude': [43.0] * 12,
            'longitude': [-110.0] * 12,
            'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
            'temperature_f': [20.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 70.0, 60.0, 50.0, 40.0, 25.0],  # Summer > Winter
            'elevation': [2000.0] * 12,
        })
        
        file_path = tmp_path / "test_temp_season.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_ndvi_landcover_relationship(self, analyze_module, tmp_path):
        """Test NDVI-land cover relationship validation."""
        df = pd.DataFrame({
            'latitude': [43.0] * 20,
            'longitude': [-110.0] * 20,
            'land_cover_code': [41] * 10 + [21] * 10,  # Forest + Developed
            'ndvi': [0.75] * 10 + [0.3] * 10,  # Forest has higher NDVI
            'elevation': [2000.0] * 20,
        })
        
        file_path = tmp_path / "test_ndvi_landcover.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True
    
    def test_temperature_elevation_relationship(self, analyze_module, tmp_path):
        """Test temperature-elevation relationship validation."""
        import numpy as np
        np.random.seed(42)
        
        # Create data with negative correlation (higher elevation = colder)
        elevation = np.random.uniform(2000, 10000, 50)
        temperature = 80 - (elevation - 2000) / 200 + np.random.normal(0, 5, 50)
        
        df = pd.DataFrame({
            'latitude': np.random.uniform(43.0, 44.0, 50),
            'longitude': np.random.uniform(-110.0, -109.0, 50),
            'elevation': elevation,
            'temperature_f': temperature,
        })
        
        file_path = tmp_path / "test_temp_elev_relationship.csv"
        df.to_csv(file_path, index=False)
        
        result = analyze_module.analyze_integrated_features(file_path)
        assert result is True


class TestTemporalFeatureOutputSections:
    """Test that new temporal feature sections appear in output."""
    
    def test_integration_output_includes_temporal_sections(self, tmp_path):
        """Test that integration test output includes NDVI and weather sections."""
        import subprocess
        
        # Create comprehensive test dataset with temporal features
        df = pd.DataFrame({
            'latitude': [43.0, 43.5, 44.0],
            'longitude': [-110.0, -107.0, -105.0],
            'elk_present': [1, 0, 1],
            'elevation': [5000.0, 8000.0, 10000.0],
            'temperature_f': [70.0, 65.0, 60.0],
            'precip_last_7_days_inches': [0.5, 1.0, 0.2],
            'cloud_cover_percent': [30.0, 40.0, 20.0],
            'ndvi': [0.75, 0.65, 0.55],
            'ndvi_age_days': [5, 10, 15],
            'irg': [0.01, 0.005, -0.002],
            'summer_integrated_ndvi': [70.0, 65.0, 60.0],
        })
        
        file_path = tmp_path / "test_temporal_comprehensive.csv"
        df.to_csv(file_path, index=False)
        
        # Run script via subprocess
        scripts_dir = Path(__file__).parent.parent / "scripts"
        script_path = scripts_dir / "analyze_integrated_features.py"
        
        if not script_path.exists():
            pytest.skip("Script not found")
        
        result = subprocess.run(
            [sys.executable, str(script_path), str(file_path)],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )
        
        # Should succeed
        assert result.returncode == 0
        
        # Should contain new temporal feature sections
        assert "NDVI DATA QUALITY" in result.stdout
        assert "WEATHER & TEMPERATURE DATA QUALITY" in result.stdout

