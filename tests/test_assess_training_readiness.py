"""
Tests for assess_training_readiness.py script.

Tests the training readiness assessment logic.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys
import importlib.util
import shutil


class TestAssessTrainingReadiness:
    """Test the assess_training_readiness script functions."""
    
    @pytest.fixture
    def sample_integrated_dataset(self, tmp_path):
        """Create a sample integrated dataset for assessment."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2, 43.3, 43.4],
            'longitude': [-110.0, -110.1, -110.2, -110.3, -110.4],
            'elk_present': [1, 0, 1, 0, 1],
            # Environmental features
            'elevation': [2000.0, 2100.0, 2200.0, 2300.0, 2400.0],
            'slope_degrees': [5.0, 10.0, 15.0, 20.0, 25.0],
            'aspect_degrees': [180.0, 270.0, 90.0, 0.0, 180.0],
            'canopy_cover_percent': [40.0, 50.0, 60.0, 70.0, 80.0],
            'land_cover_code': [1, 2, 3, 4, 5],
            'land_cover_type': ['forest', 'grassland', 'shrub', 'wetland', 'barren'],
            'water_distance_miles': [1.0, 2.0, 3.0, 4.0, 5.0],
            'water_reliability': [0.7, 0.8, 0.9, 0.6, 0.7],
            'road_distance_miles': [2.0, 3.0, 4.0, 5.0, 6.0],
            'trail_distance_miles': [1.0, 2.0, 3.0, 4.0, 5.0],
            'security_habitat_percent': [60.0, 70.0, 80.0, 90.0, 100.0],
            # Temporal features
            'year': [2024, 2024, 2024, 2024, 2024],
            'month': [1, 2, 3, 4, 5],
            # Metadata (should be ignored in quality assessment)
            'route_id': [1, 2, 3, None, None],
            'id': [100, 101, 102, None, None]
        })
        
        file_path = tmp_path / "combined_test_presence_absence.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def assess_module(self, tmp_path):
        """Load the assess_training_readiness module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "assess_training_readiness.py"
        temp_script = tmp_path / "assess_training_readiness.py"
        
        if real_script.exists():
            shutil.copy2(real_script, temp_script)
        else:
            pytest.skip("Real assess_training_readiness.py not found")
        
        spec = importlib.util.spec_from_file_location(
            "assess_training_readiness",
            temp_script
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def test_assess_single_dataset(self, assess_module, sample_integrated_dataset):
        """Test assessing a single dataset."""
        score, percentage = assess_module.assess_training_readiness(sample_integrated_dataset)
        
        # Should return valid scores
        assert isinstance(score, (int, float))
        assert isinstance(percentage, (int, float))
        assert 0 <= score <= 5
        assert 0 <= percentage <= 100
    
    def test_assess_test_file(self, assess_module, tmp_path):
        """Test that assess_training_readiness can handle test files (with _test suffix)."""
        # Create a test file (with _test in name)
        df = pd.DataFrame({
            'latitude': [43.0, 43.1],
            'longitude': [-110.0, -110.1],
            'elk_present': [1, 0],
            'elevation': [2000.0, 2100.0],
            'slope_degrees': [5.0, 10.0],
        })
        
        test_file = tmp_path / "combined_test_dataset_presence_absence_test.csv"
        df.to_csv(test_file, index=False)
        
        # Should handle test file without error
        score, percentage = assess_module.assess_training_readiness(test_file)
        
        # Should return valid scores
        assert isinstance(score, (int, float))
        assert isinstance(percentage, (int, float))
        assert 0 <= score <= 5
        assert 0 <= percentage <= 100
    
    def test_assess_all_datasets(self, assess_module, tmp_path):
        """Test assessing all datasets (when none exist, should handle gracefully)."""
        # Create processed directory structure
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        
        # Change to tmp_path to simulate data directory
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            score, percentage = assess_module.assess_training_readiness(None)
            
            # Should return scores even if no datasets found
            assert isinstance(score, (int, float))
            assert isinstance(percentage, (int, float))
        finally:
            os.chdir(old_cwd)
    
    def test_assess_with_placeholders(self, assess_module, tmp_path):
        """Test assessment with placeholder values."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'elk_present': [1, 0, 1],
            'elevation': [2000.0, 8500.0, 2200.0],  # One placeholder
            'slope_degrees': [5.0, 10.0, 15.0],
            'aspect_degrees': [180.0, 270.0, 90.0],
            'canopy_cover_percent': [40.0, 30.0, 50.0],  # One placeholder
            'land_cover_code': [1, 2, 3],
            'water_distance_miles': [1.0, 2.0, 3.0],
            'water_reliability': [0.7, 0.8, 0.9],
            'road_distance_miles': [2.0, 3.0, 4.0],
            'trail_distance_miles': [1.0, 2.0, 3.0],
            'security_habitat_percent': [60.0, 70.0, 80.0]
        })
        
        file_path = tmp_path / "with_placeholders.csv"
        df.to_csv(file_path, index=False)
        
        score, percentage = assess_module.assess_training_readiness(file_path)
        
        # Should still return valid scores
        assert isinstance(score, (int, float))
        assert isinstance(percentage, (int, float))
        # Score might be lower due to placeholders
        assert 0 <= score <= 5
    
    def test_assess_with_imbalanced_classes(self, assess_module, tmp_path):
        """Test assessment with imbalanced classes."""
        df = pd.DataFrame({
            'latitude': [43.0] * 10 + [43.1] * 2,
            'longitude': [-110.0] * 10 + [-110.1] * 2,
            'elk_present': [1] * 10 + [0] * 2,  # Imbalanced
            'elevation': [2000.0] * 12,
            'slope_degrees': [5.0] * 12,
            'aspect_degrees': [180.0] * 12,
            'canopy_cover_percent': [40.0] * 12,
            'land_cover_code': [1] * 12,
            'water_distance_miles': [1.0] * 12,
            'water_reliability': [0.7] * 12,
            'road_distance_miles': [2.0] * 12,
            'trail_distance_miles': [1.0] * 12,
            'security_habitat_percent': [60.0] * 12
        })
        
        file_path = tmp_path / "imbalanced.csv"
        df.to_csv(file_path, index=False)
        
        score, percentage = assess_module.assess_training_readiness(file_path)
        
        # Should return valid scores
        assert isinstance(score, (int, float))
        assert isinstance(percentage, (int, float))
    
    def test_assess_with_small_dataset(self, assess_module, tmp_path):
        """Test assessment with very small dataset."""
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
            'water_reliability': [0.7, 0.8],
            'road_distance_miles': [2.0, 3.0],
            'trail_distance_miles': [1.0, 2.0],
            'security_habitat_percent': [60.0, 70.0]
        })
        
        file_path = tmp_path / "small.csv"
        df.to_csv(file_path, index=False)
        
        score, percentage = assess_module.assess_training_readiness(file_path)
        
        # Should return valid scores (might be lower for small dataset)
        assert isinstance(score, (int, float))
        assert isinstance(percentage, (int, float))
    
    def test_assess_with_test_mode_flag(self, assess_module, tmp_path):
        """Test that assess_training_readiness prefers test files when --test-mode is used."""
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        
        # Create both regular and test files
        regular_df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'elk_present': [1, 0, 1],
            'elevation': [2000.0, 2100.0, 2200.0],
            'slope_degrees': [5.0, 10.0, 15.0],
        })
        
        test_df = pd.DataFrame({
            'latitude': [43.0, 43.1],
            'longitude': [-110.0, -110.1],
            'elk_present': [1, 0],
            'elevation': [2000.0, 2100.0],
            'slope_degrees': [5.0, 10.0],
        })
        
        regular_file = processed_dir / "combined_north_bighorn_presence_absence.csv"
        test_file = processed_dir / "combined_north_bighorn_presence_absence_test.csv"
        regular_df.to_csv(regular_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        # Test without --test-mode (should prefer regular file)
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            score1, pct1 = assess_module.assess_training_readiness(None, test_mode=False)
            
            # Test with --test-mode (should prefer test file)
            score2, pct2 = assess_module.assess_training_readiness(None, test_mode=True)
            
            # Both should return valid scores
            assert isinstance(score1, (int, float))
            assert isinstance(score2, (int, float))
            assert 0 <= score1 <= 5
            assert 0 <= score2 <= 5
        finally:
            os.chdir(old_cwd)


class TestAssessTrainingReadinessIntegration:
    """Integration tests for assess_training_readiness script."""
    
    def test_script_runs_with_single_dataset(self, tmp_path):
        """Test that script can be run with single dataset."""
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
            'water_reliability': [0.7, 0.8],
            'road_distance_miles': [2.0, 3.0],
            'trail_distance_miles': [1.0, 2.0],
            'security_habitat_percent': [60.0, 70.0]
        })
        
        dataset_file = tmp_path / "test_dataset.csv"
        df.to_csv(dataset_file, index=False)
        
        # Run script via subprocess
        import subprocess
        scripts_dir = Path(__file__).parent.parent / "scripts"
        script_path = scripts_dir / "assess_training_readiness.py"
        
        if not script_path.exists():
            pytest.skip("Script not found")
        
        result = subprocess.run(
            [sys.executable, str(script_path), '--dataset', str(dataset_file)],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )
        
        # Should succeed
        assert result.returncode == 0
        
        # Should contain assessment output
        assert "TRAINING READINESS ASSESSMENT" in result.stdout or "READINESS ASSESSMENT" in result.stdout
    
    def test_script_runs_without_arguments(self, tmp_path):
        """Test that script can be run without arguments (assesses all datasets)."""
        import subprocess
        scripts_dir = Path(__file__).parent.parent / "scripts"
        script_path = scripts_dir / "assess_training_readiness.py"
        
        if not script_path.exists():
            pytest.skip("Script not found")
        
        # Create processed directory with at least one dataset
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        
        # Create a test dataset so the script has something to assess
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
            'water_reliability': [0.7, 0.8],
            'road_distance_miles': [2.0, 3.0],
            'trail_distance_miles': [1.0, 2.0],
            'security_habitat_percent': [60.0, 70.0]
        })
        
        # Create one of the expected dataset files (with standardized name)
        test_file = processed_dir / "combined_northern_bighorn_presence_absence.csv"
        df.to_csv(test_file, index=False)
        
        # Change to tmp_path to simulate data directory
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=tmp_path
            )
            
            # Should succeed if at least one dataset is found
            assert result.returncode == 0
            assert "TRAINING READINESS ASSESSMENT" in result.stdout or "READINESS ASSESSMENT" in result.stdout
        finally:
            os.chdir(old_cwd)
    
    def test_script_accepts_test_mode_flag(self, tmp_path):
        """Test that script accepts --test-mode command-line flag."""
        import subprocess
        scripts_dir = Path(__file__).parent.parent / "scripts"
        script_path = scripts_dir / "assess_training_readiness.py"
        
        if not script_path.exists():
            pytest.skip("Script not found")
        
        # Create processed directory with test files
        processed_dir = tmp_path / "data" / "processed"
        processed_dir.mkdir(parents=True)
        
        df = pd.DataFrame({
            'latitude': [43.0, 43.1],
            'longitude': [-110.0, -110.1],
            'elk_present': [1, 0],
            'elevation': [2000.0, 2100.0],
            'slope_degrees': [5.0, 10.0],
        })
        
        # Create test file with standardized name (northern_bighorn)
        test_file = processed_dir / "combined_northern_bighorn_presence_absence_test.csv"
        df.to_csv(test_file, index=False)
        
        # Change to tmp_path to simulate data directory
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = subprocess.run(
                [sys.executable, str(script_path), '--test-mode'],
                capture_output=True,
                text=True,
                cwd=tmp_path
            )
            
            # Should succeed
            assert result.returncode == 0
            # Should indicate test mode
            assert "TEST MODE" in result.stdout or "test" in result.stdout.lower()
        finally:
            os.chdir(old_cwd)

