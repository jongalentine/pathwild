"""
Tests for prepare_training_features.py script.

Tests the feature preparation logic that excludes metadata columns.
"""
import pytest
import pandas as pd
from pathlib import Path
import sys
import importlib.util
import shutil

# Add scripts to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


class TestPrepareTrainingFeatures:
    """Test the prepare_training_features script functions."""
    
    @pytest.fixture
    def sample_combined_dataset(self, tmp_path):
        """Create a sample combined dataset with metadata and features."""
        df = pd.DataFrame({
            'latitude': [43.0, 43.1, 43.2],
            'longitude': [-110.0, -110.1, -110.2],
            'elk_present': [1, 0, 1],
            # Metadata columns (should be excluded)
            'route_id': [1, None, 2],
            'id': [100, 101, 102],
            'Elk_ID': ['E1', 'E2', 'E3'],
            'point_index': [0, None, 1],  # Route-specific metadata
            'absence_strategy': [None, 'background', None],
            'distance_to_area_048_km': [10.5, 11.2, 9.8],
            'firstdate': ['2024-01-01', None, '2024-01-02'],
            # Predator/wildlife columns (temporarily excluded)
            'wolf_data_quality': [0.75, 0.5, 0.75],
            'bear_data_quality': [0.65, 0.5, 0.65],
            'bear_activity_distance_miles': [5.0, 5.0, 5.0],
            'wolves_per_1000_elk': [2.0, 2.0, 2.0],
            'pregnancy_rate': [0.85, 0.85, 0.85],
            # Environmental features (should be kept)
            'elevation': [2000.0, 2100.0, 2200.0],
            'slope_degrees': [5.0, 10.0, 15.0],
            'aspect_degrees': [180.0, 270.0, 90.0],
            'canopy_cover_percent': [30.0, 40.0, 50.0],
            'land_cover_code': [1, 2, 3],
            'water_distance_miles': [0.5, 1.0, 1.5],
            'water_reliability': [0.8, 0.7, 0.9],
            'road_distance_miles': [2.0, 3.0, 4.0],
            'trail_distance_miles': [1.0, 2.0, 3.0],
            'security_habitat_percent': [60.0, 70.0, 80.0],
            # Temporal columns (optional)
            'year': [2024, 2024, 2024],
            'month': [1, 2, 3]
        })
        
        file_path = tmp_path / "combined_test_presence_absence.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def prepare_module(self, tmp_path):
        """Load the prepare_training_features module."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "prepare_training_features.py"
        temp_script = tmp_path / "prepare_training_features.py"
        
        if real_script.exists():
            shutil.copy2(real_script, temp_script)
        else:
            pytest.skip("Real prepare_training_features.py not found")
        
        spec = importlib.util.spec_from_file_location(
            "prepare_training_features",
            temp_script
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def test_get_feature_columns_excludes_metadata(self, prepare_module, sample_combined_dataset):
        """Test that get_feature_columns excludes metadata columns."""
        df = pd.read_csv(sample_combined_dataset)
        
        feature_cols = prepare_module.get_feature_columns(df)
        
        # Should exclude metadata
        assert 'route_id' not in feature_cols
        assert 'id' not in feature_cols
        assert 'Elk_ID' not in feature_cols
        assert 'point_index' not in feature_cols  # Route-specific metadata
        assert 'absence_strategy' not in feature_cols
        assert 'distance_to_area_048_km' not in feature_cols
        assert 'firstdate' not in feature_cols
        # Should exclude predator/wildlife columns (temporarily excluded)
        assert 'wolf_data_quality' not in feature_cols
        assert 'bear_data_quality' not in feature_cols
        assert 'bear_activity_distance_miles' not in feature_cols
        assert 'wolves_per_1000_elk' not in feature_cols
        assert 'pregnancy_rate' not in feature_cols
        
        # Should include environmental features
        assert 'elevation' in feature_cols
        assert 'slope_degrees' in feature_cols
        assert 'aspect_degrees' in feature_cols
        assert 'canopy_cover_percent' in feature_cols
        assert 'water_distance_miles' in feature_cols
        
        # Should include target (elk_present) but not as a feature
        assert 'elk_present' not in feature_cols  # It's added separately
    
    def test_get_feature_columns_includes_temporal_by_default(self, prepare_module, sample_combined_dataset):
        """Test that temporal columns are included by default."""
        df = pd.read_csv(sample_combined_dataset)
        
        feature_cols = prepare_module.get_feature_columns(df, exclude_temporal=False)
        
        assert 'year' in feature_cols
        assert 'month' in feature_cols
    
    def test_get_feature_columns_excludes_temporal_when_requested(self, prepare_module, sample_combined_dataset):
        """Test that temporal columns can be excluded."""
        df = pd.read_csv(sample_combined_dataset)
        
        feature_cols = prepare_module.get_feature_columns(df, exclude_temporal=True)
        
        assert 'year' not in feature_cols
        assert 'month' not in feature_cols
    
    def test_prepare_training_dataset_excludes_metadata(self, prepare_module, sample_combined_dataset, tmp_path):
        """Test that prepare_training_dataset creates clean feature dataset."""
        output_file = tmp_path / "test_features.csv"
        
        df_features = prepare_module.prepare_training_dataset(
            sample_combined_dataset,
            output_file,
            exclude_temporal=False
        )
        
        # Check output file exists
        assert output_file.exists()
        
        # Check columns
        assert 'route_id' not in df_features.columns
        assert 'id' not in df_features.columns
        assert 'Elk_ID' not in df_features.columns
        assert 'point_index' not in df_features.columns
        assert 'absence_strategy' not in df_features.columns
        # Check predator/wildlife columns are excluded
        assert 'wolf_data_quality' not in df_features.columns
        assert 'bear_data_quality' not in df_features.columns
        assert 'bear_activity_distance_miles' not in df_features.columns
        assert 'wolves_per_1000_elk' not in df_features.columns
        assert 'pregnancy_rate' not in df_features.columns
        
        # Check features are present
        assert 'elevation' in df_features.columns
        assert 'slope_degrees' in df_features.columns
        assert 'elk_present' in df_features.columns  # Target should be included
        
        # Check data integrity
        assert len(df_features) == 3
        assert df_features['elevation'].notna().all()
    
    def test_prepare_training_dataset_preserves_target(self, prepare_module, sample_combined_dataset, tmp_path):
        """Test that target variable is preserved."""
        output_file = tmp_path / "test_features.csv"
        
        df_features = prepare_module.prepare_training_dataset(
            sample_combined_dataset,
            output_file,
            exclude_temporal=False
        )
        
        # Target should be first column
        assert df_features.columns[0] == 'elk_present'
        
        # Target values should be preserved
        assert df_features['elk_present'].tolist() == [1, 0, 1]
    
    def test_prepare_training_dataset_excludes_temporal(self, prepare_module, sample_combined_dataset, tmp_path):
        """Test that temporal columns can be excluded."""
        output_file = tmp_path / "test_features.csv"
        
        df_features = prepare_module.prepare_training_dataset(
            sample_combined_dataset,
            output_file,
            exclude_temporal=True
        )
        
        assert 'year' not in df_features.columns
        assert 'month' not in df_features.columns
    
    def test_prepare_all_datasets(self, prepare_module, tmp_path):
        """Test prepare_all_datasets function."""
        processed_dir = tmp_path / "processed"
        features_dir = tmp_path / "features"
        processed_dir.mkdir()
        features_dir.mkdir()
        
        # Create combine_feature_files.py script in the same directory as prepare_training_features
        # This is needed for the import to work
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_combine_script = scripts_dir / "combine_feature_files.py"
        
        # Copy combine script to tmp_path so it can be found
        if real_combine_script.exists():
            temp_combine_script = tmp_path / "combine_feature_files.py"
            shutil.copy2(real_combine_script, temp_combine_script)
        
        # Create multiple combined datasets
        for dataset_name in ['north_bighorn', 'south_bighorn']:
            df = pd.DataFrame({
                'latitude': [43.0, 43.1],
                'longitude': [-110.0, -110.1],
                'elk_present': [1, 0],
                'route_id': [1, 2],  # Metadata
                'elevation': [2000.0, 2100.0],  # Feature
                'slope_degrees': [5.0, 10.0]  # Feature
            })
            combined_file = processed_dir / f"combined_{dataset_name}_presence_absence.csv"
            df.to_csv(combined_file, index=False)
        
        # Run prepare_all_datasets
        prepare_module.prepare_all_datasets(
            processed_dir=processed_dir,
            features_dir=features_dir,
            exclude_temporal=False
        )
        
        # Check that feature files were created
        assert (features_dir / "north_bighorn_features.csv").exists()
        assert (features_dir / "south_bighorn_features.csv").exists()
        
        # Check that metadata was excluded
        df_north = pd.read_csv(features_dir / "north_bighorn_features.csv")
        assert 'route_id' not in df_north.columns
        assert 'elevation' in df_north.columns
        
        # Check that complete_context.csv was created (if combine script exists)
        complete_context_file = features_dir / "complete_context.csv"
        complete_context_test_file = features_dir / "complete_context_test.csv"
        if real_combine_script.exists():
            assert complete_context_file.exists(), "complete_context.csv should be created"
            assert not complete_context_test_file.exists(), "complete_context_test.csv should not be created in normal mode"
            df_complete = pd.read_csv(complete_context_file)
            # Should have combined rows from both datasets
            assert len(df_complete) == 4  # 2 rows from each dataset
            assert 'elk_present' in df_complete.columns
            assert 'elevation' in df_complete.columns
            assert 'route_id' not in df_complete.columns
    
    def test_prepare_all_datasets_test_mode(self, prepare_module, tmp_path):
        """Test prepare_all_datasets function in test mode (with limit)."""
        processed_dir = tmp_path / "processed"
        features_dir = tmp_path / "features"
        processed_dir.mkdir()
        features_dir.mkdir()
        
        # Create combine_feature_files.py script in the same directory as prepare_training_features
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_combine_script = scripts_dir / "combine_feature_files.py"
        
        # Copy combine script to tmp_path so it can be found
        if real_combine_script.exists():
            temp_combine_script = tmp_path / "combine_feature_files.py"
            shutil.copy2(real_combine_script, temp_combine_script)
        
        # Create multiple test combined datasets
        for dataset_name in ['north_bighorn', 'south_bighorn']:
            df = pd.DataFrame({
                'latitude': [43.0, 43.1],
                'longitude': [-110.0, -110.1],
                'elk_present': [1, 0],
                'route_id': [1, 2],  # Metadata
                'elevation': [2000.0, 2100.0],  # Feature
                'slope_degrees': [5.0, 10.0]  # Feature
            })
            combined_file = processed_dir / f"combined_{dataset_name}_presence_absence_test.csv"
            df.to_csv(combined_file, index=False)
        
        # Run prepare_all_datasets in test mode
        prepare_module.prepare_all_datasets(
            processed_dir=processed_dir,
            features_dir=features_dir,
            exclude_temporal=False,
            limit=100  # Test mode
        )
        
        # Check that test feature files were created
        assert (features_dir / "north_bighorn_features_test.csv").exists()
        assert (features_dir / "south_bighorn_features_test.csv").exists()
        
        # Check that metadata was excluded
        df_north = pd.read_csv(features_dir / "north_bighorn_features_test.csv")
        assert 'route_id' not in df_north.columns
        assert 'elevation' in df_north.columns
        
        # Check that complete_context_test.csv was created (if combine script exists)
        complete_context_file = features_dir / "complete_context.csv"
        complete_context_test_file = features_dir / "complete_context_test.csv"
        if real_combine_script.exists():
            assert not complete_context_file.exists(), "complete_context.csv should not be created in test mode"
            assert complete_context_test_file.exists(), "complete_context_test.csv should be created in test mode"
            df_complete = pd.read_csv(complete_context_test_file)
            # Should have combined rows from both test datasets
            assert len(df_complete) == 4  # 2 rows from each dataset
            assert 'elk_present' in df_complete.columns
            assert 'elevation' in df_complete.columns
            assert 'route_id' not in df_complete.columns


class TestPrepareTrainingFeaturesIntegration:
    """Integration tests for prepare_training_features script."""
    
    def test_script_runs_with_single_dataset(self, tmp_path):
        """Test that script can be run with single dataset."""
        processed_dir = tmp_path / "processed"
        features_dir = tmp_path / "features"
        processed_dir.mkdir(parents=True)
        features_dir.mkdir(parents=True)
        
        # Create combined dataset
        df = pd.DataFrame({
            'latitude': [43.0, 43.1],
            'longitude': [-110.0, -110.1],
            'elk_present': [1, 0],
            'route_id': [1, 2],
            'elevation': [2000.0, 2100.0],
            'slope_degrees': [5.0, 10.0]
        })
        combined_file = processed_dir / "combined_test_presence_absence.csv"
        df.to_csv(combined_file, index=False)
        
        output_file = features_dir / "test_features.csv"
        
        # Run script via subprocess
        import subprocess
        scripts_dir = Path(__file__).parent.parent / "scripts"
        script_path = scripts_dir / "prepare_training_features.py"
        
        if not script_path.exists():
            pytest.skip("Script not found")
        
        result = subprocess.run(
            [sys.executable, str(script_path), str(combined_file), str(output_file)],
            capture_output=True,
            text=True,
            cwd=tmp_path
        )
        
        # Should succeed
        assert result.returncode == 0
        
        # Check output file
        assert output_file.exists()
        df_output = pd.read_csv(output_file)
        assert 'route_id' not in df_output.columns
        assert 'elevation' in df_output.columns
        assert 'elk_present' in df_output.columns


class TestCombineFeatureFiles:
    """Tests for combine_feature_files.py script."""
    
    def test_combine_feature_files(self, tmp_path):
        """Test that combine_feature_files combines multiple feature files."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        # Create multiple feature files
        for i, dataset_name in enumerate(['north_bighorn', 'south_bighorn']):
            df = pd.DataFrame({
                'elk_present': [1, 0],
                'latitude': [43.0 + i, 43.1 + i],
                'longitude': [-110.0, -110.1],
                'elevation': [2000.0 + i * 100, 2100.0 + i * 100],
                'slope_degrees': [5.0, 10.0]
            })
            feature_file = features_dir / f"{dataset_name}_features.csv"
            df.to_csv(feature_file, index=False)
        
        # Import combine module
        scripts_dir = Path(__file__).parent.parent / "scripts"
        combine_script = scripts_dir / "combine_feature_files.py"
        
        if not combine_script.exists():
            pytest.skip("combine_feature_files.py not found")
        
        spec = importlib.util.spec_from_file_location(
            "combine_feature_files",
            combine_script
        )
        combine_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(combine_module)
        
        # Run combine
        output_file = features_dir / "complete_context.csv"
        combined_df = combine_module.combine_feature_files(
            features_dir=features_dir,
            output_file=output_file,
            exclude_test_files=True
        )
        
        # Check output
        assert output_file.exists()
        assert len(combined_df) == 4  # 2 rows from each dataset
        assert 'elk_present' in combined_df.columns
        assert 'elevation' in combined_df.columns
        assert 'latitude' in combined_df.columns
        
        # Verify all data is present
        assert combined_df['elk_present'].sum() == 2  # Two 1s and two 0s
        assert len(combined_df[combined_df['elk_present'] == 0]) == 2
    
    def test_combine_feature_files_excludes_test_files(self, tmp_path):
        """Test that test files are excluded by default."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        # Create regular and test feature files
        df_regular = pd.DataFrame({
            'elk_present': [1, 0],
            'elevation': [2000.0, 2100.0]
        })
        df_regular.to_csv(features_dir / "north_bighorn_features.csv", index=False)
        
        df_test = pd.DataFrame({
            'elk_present': [1, 1, 1],
            'elevation': [3000.0, 3100.0, 3200.0]
        })
        df_test.to_csv(features_dir / "north_bighorn_features_test.csv", index=False)
        
        # Import and run combine
        scripts_dir = Path(__file__).parent.parent / "scripts"
        combine_script = scripts_dir / "combine_feature_files.py"
        
        if not combine_script.exists():
            pytest.skip("combine_feature_files.py not found")
        
        spec = importlib.util.spec_from_file_location(
            "combine_feature_files",
            combine_script
        )
        combine_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(combine_module)
        
        output_file = features_dir / "complete_context.csv"
        combined_df = combine_module.combine_feature_files(
            features_dir=features_dir,
            output_file=output_file,
            exclude_test_files=True
        )
        
        # Should only have rows from regular file (not test file)
        assert len(combined_df) == 2  # Only from regular file
    
    def test_combine_feature_files_test_mode(self, tmp_path):
        """Test that combine_feature_files works in test mode."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        # Create test feature files only
        for i, dataset_name in enumerate(['north_bighorn', 'south_bighorn']):
            df = pd.DataFrame({
                'elk_present': [1, 0],
                'latitude': [43.0 + i, 43.1 + i],
                'longitude': [-110.0, -110.1],
                'elevation': [2000.0 + i * 100, 2100.0 + i * 100],
                'slope_degrees': [5.0, 10.0]
            })
            feature_file = features_dir / f"{dataset_name}_features_test.csv"
            df.to_csv(feature_file, index=False)
        
        # Import combine module
        scripts_dir = Path(__file__).parent.parent / "scripts"
        combine_script = scripts_dir / "combine_feature_files.py"
        
        if not combine_script.exists():
            pytest.skip("combine_feature_files.py not found")
        
        spec = importlib.util.spec_from_file_location(
            "combine_feature_files",
            combine_script
        )
        combine_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(combine_module)
        
        # Run combine in test mode
        output_file = features_dir / "complete_context_test.csv"
        combined_df = combine_module.combine_feature_files(
            features_dir=features_dir,
            output_file=output_file,
            exclude_test_files=True,
            test_mode=True
        )
        
        # Check output
        assert output_file.exists()
        assert len(combined_df) == 4  # 2 rows from each dataset
        assert 'elk_present' in combined_df.columns
        assert 'elevation' in combined_df.columns
        assert 'latitude' in combined_df.columns
        
        # Verify all data is present
        assert combined_df['elk_present'].sum() == 2  # Two 1s and two 0s
        assert len(combined_df[combined_df['elk_present'] == 0]) == 2

