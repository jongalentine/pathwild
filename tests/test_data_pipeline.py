"""
Tests for automated data processing pipeline.

Tests the pipeline orchestrator and individual pipeline steps.
"""
import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess
import sys
import shutil

# Import pipeline components
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import importlib.util

scripts_dir = Path(__file__).parent.parent / "scripts"
spec = importlib.util.spec_from_file_location(
    "run_data_pipeline",
    scripts_dir / "run_data_pipeline.py"
)
pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_module)

PipelineStep = pipeline_module.PipelineStep
DataPipeline = pipeline_module.DataPipeline


class TestPipelineStep:
    """Test individual pipeline steps."""
    
    def test_step_initialization(self, tmp_path):
        """Test pipeline step initialization."""
        script_path = tmp_path / "test_script.py"
        script_path.write_text("print('test')")
        
        step = PipelineStep(
            name='test_step',
            description='Test step',
            script_path=script_path,
            command_args=['--arg', 'value'],
            expected_output=tmp_path / "output.csv"
        )
        
        assert step.name == 'test_step'
        assert step.description == 'Test step'
        assert step.script_path == script_path
        assert step.command_args == ['--arg', 'value']
    
    def test_step_should_skip(self, tmp_path):
        """Test step skipping logic."""
        script_path = tmp_path / "test_script.py"
        script_path.write_text("print('test')")
        
        step = PipelineStep(
            name='test_step',
            description='Test step',
            script_path=script_path,
            command_args=[]
        )
        
        assert step.should_skip(['test_step']) is True
        assert step.should_skip(['other_step']) is False
        assert step.should_skip([]) is False
    
    def test_step_can_run(self, tmp_path):
        """Test step can_run logic."""
        script_path = tmp_path / "test_script.py"
        script_path.write_text("print('test')")
        
        # Step with existing script and input
        input_file = tmp_path / "input.csv"
        input_file.write_text("test")
        
        step = PipelineStep(
            name='test_step',
            description='Test step',
            script_path=script_path,
            command_args=[],
            required_input=input_file
        )
        
        assert step.can_run() is True
        
        # Step with missing script
        step2 = PipelineStep(
            name='test_step2',
            description='Test step',
            script_path=tmp_path / "missing.py",
            command_args=[]
        )
        
        assert step2.can_run() is False
    
    def test_step_is_complete(self, tmp_path):
        """Test step completion checking."""
        script_path = tmp_path / "test_script.py"
        script_path.write_text("print('test')")
        output_file = tmp_path / "output.csv"
        output_file.write_text("test")
        
        step = PipelineStep(
            name='test_step',
            description='Test step',
            script_path=script_path,
            command_args=[],
            expected_output=output_file
        )
        
        assert step.is_complete() is True
        
        # Step without output file
        step2 = PipelineStep(
            name='test_step2',
            description='Test step',
            script_path=script_path,
            command_args=[],
            expected_output=tmp_path / "missing.csv"
        )
        
        assert step2.is_complete() is False


class TestDataPipeline:
    """Test the data pipeline orchestrator."""
    
    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create a test pipeline."""
        # Create mock data directory structure
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"
        raw_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)
        
        # Create a test dataset directory
        (raw_dir / "elk_test_dataset").mkdir()
        (raw_dir / "elk_test_dataset" / "test_data.csv").write_text(
            "latitude,longitude\n43.0,-110.0\n43.1,-110.1"
        )
        
        # Create pipeline - it will use real scripts directory
        # But we'll patch the script paths in tests to avoid overwriting real files
        pipeline = DataPipeline(
            data_dir=data_dir,
            dataset_name='test_dataset',
            skip_steps=[],
            force=False
        )
        
        # Patch script paths to point to temporary files to avoid overwriting real scripts
        temp_scripts_dir = tmp_path / "scripts"
        temp_scripts_dir.mkdir()
        for step in pipeline.steps:
            if step.script_path:
                # Always create a temporary script path, even if real one doesn't exist
                temp_script_path = temp_scripts_dir / step.script_path.name
                # Try to copy the real script if it exists, otherwise create a mock
                real_script = step.script_path
                if real_script.exists():
                    shutil.copy2(real_script, temp_script_path)
                else:
                    temp_script_path.write_text("#!/usr/bin/env python3\nprint('mock script')")
                temp_script_path.chmod(0o755)
                # Replace the script_path with the temporary one
                step.script_path = temp_script_path
        
        return pipeline
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.data_dir.exists()
        assert pipeline.dataset_name == 'test_dataset'
        assert len(pipeline.steps) > 0
    
    def test_pipeline_builds_steps(self, pipeline):
        """Test that pipeline builds correct steps."""
        step_names = [step.name for step in pipeline.steps]
        
        # Should have all expected steps
        assert 'process_raw' in step_names
        assert 'generate_absence' in step_names
        assert 'integrate_features' in step_names
        assert 'analyze_features' in step_names
        assert 'assess_readiness' in step_names
    
    @patch('subprocess.run')
    def test_pipeline_run_success(self, mock_run, pipeline):
        """Test successful pipeline run."""
        # Mock successful subprocess runs
        mock_run.return_value = Mock(returncode=0)
        
        # Create mock prerequisite files so prerequisite check passes
        data_dir = pipeline.data_dir
        (data_dir / 'dem').mkdir(parents=True, exist_ok=True)
        (data_dir / 'dem' / 'wyoming_dem.tif').touch()
        (data_dir / 'terrain').mkdir(parents=True, exist_ok=True)
        (data_dir / 'terrain' / 'slope.tif').touch()
        (data_dir / 'terrain' / 'aspect.tif').touch()
        (data_dir / 'landcover').mkdir(parents=True, exist_ok=True)
        (data_dir / 'landcover' / 'nlcd.tif').touch()
        (data_dir / 'canopy').mkdir(parents=True, exist_ok=True)
        (data_dir / 'canopy' / 'canopy_cover.tif').touch()
        (data_dir / 'hydrology').mkdir(parents=True, exist_ok=True)
        (data_dir / 'hydrology' / 'water_sources.geojson').write_text('{"type": "FeatureCollection", "features": []}')
        
        # Ensure script paths exist (they should already be patched to temp files in fixture)
        for step in pipeline.steps:
            if step.script_path and not step.script_path.exists():
                step.script_path.parent.mkdir(parents=True, exist_ok=True)
                step.script_path.write_text("#!/usr/bin/env python3\nprint('mock script')")
            if step.required_input:
                # Check if required_input is a directory or file
                if step.required_input.suffix == '' or step.required_input.name.endswith('_dataset'):
                    # It's a directory
                    step.required_input.mkdir(parents=True, exist_ok=True)
                else:
                    # It's a file
                    step.required_input.parent.mkdir(parents=True, exist_ok=True)
                    step.required_input.write_text("test")
        
        # Run pipeline
        success = pipeline.run()
        
        # Should have attempted to run steps
        assert mock_run.call_count > 0
    
    def test_pipeline_skips_steps(self, pipeline):
        """Test that pipeline skips requested steps."""
        pipeline.skip_steps = ['process_raw', 'generate_absence']
        
        for step in pipeline.steps:
            if step.name in pipeline.skip_steps:
                assert step.should_skip(pipeline.skip_steps) is True
    
    def test_pipeline_includes_prepare_features_step(self, tmp_path):
        """Test that pipeline includes prepare_features step."""
        data_dir = tmp_path / "data"
        processed_dir = data_dir / "processed"
        features_dir = data_dir / "features"
        processed_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Create pipeline with dataset name
        pipeline = DataPipeline(
            data_dir=data_dir,
            dataset_name='test_dataset',
            skip_steps=[],
            force=False
        )
        
        # Check that prepare_features step exists
        step_names = [step.name for step in pipeline.steps]
        assert 'prepare_features' in step_names
        
        # Find the prepare_features step
        prepare_step = next((s for s in pipeline.steps if s.name == 'prepare_features'), None)
        assert prepare_step is not None
        assert 'prepare_training_features.py' in str(prepare_step.script_path)
    
    def test_prepare_features_step_single_dataset(self, tmp_path):
        """Test prepare_features step configuration for single dataset."""
        data_dir = tmp_path / "data"
        processed_dir = data_dir / "processed"
        features_dir = data_dir / "features"
        processed_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Create combined dataset file
        combined_file = processed_dir / "combined_test_dataset_presence_absence.csv"
        combined_file.write_text("latitude,longitude,elk_present,route_id,elevation\n43.0,-110.0,1,1,2000.0\n")
        
        pipeline = DataPipeline(
            data_dir=data_dir,
            dataset_name='test_dataset',
            skip_steps=[],
            force=False
        )
        
        # Find prepare_features step
        prepare_step = next((s for s in pipeline.steps if s.name == 'prepare_features'), None)
        assert prepare_step is not None
        
        # Check expected output
        expected_output = features_dir / "test_dataset_features.csv"
        assert prepare_step.expected_output == expected_output
        
        # Check required input
        assert prepare_step.required_input == combined_file
        
        # Check command args
        assert str(combined_file) in prepare_step.command_args
        assert str(expected_output) in prepare_step.command_args
    
    def test_prepare_features_step_all_datasets(self, tmp_path):
        """Test prepare_features step configuration for all datasets."""
        data_dir = tmp_path / "data"
        processed_dir = data_dir / "processed"
        features_dir = data_dir / "features"
        processed_dir.mkdir(parents=True, exist_ok=True)
        features_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline = DataPipeline(
            data_dir=data_dir,
            dataset_name=None,  # All datasets
            skip_steps=[],
            force=False
        )
        
        # Find prepare_features step
        prepare_step = next((s for s in pipeline.steps if s.name == 'prepare_features'), None)
        assert prepare_step is not None
        
        # Check command args include --all-datasets
        assert '--all-datasets' in prepare_step.command_args
        assert '--processed-dir' in prepare_step.command_args
        assert '--features-dir' in prepare_step.command_args
    
    def test_prepare_features_step_can_be_skipped(self, tmp_path):
        """Test that prepare_features step can be skipped."""
        data_dir = tmp_path / "data"
        processed_dir = data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline = DataPipeline(
            data_dir=data_dir,
            dataset_name='test_dataset',
            skip_steps=['prepare_features'],
            force=False
        )
        
        # Find prepare_features step
        prepare_step = next((s for s in pipeline.steps if s.name == 'prepare_features'), None)
        assert prepare_step is not None
        
        # Should be marked for skipping
        assert prepare_step.should_skip(pipeline.skip_steps) is True


class TestProcessRawPresenceData:
    """Test raw presence data processing."""
    
    @pytest.fixture
    def sample_csv_data(self, tmp_path):
        """Create sample CSV data."""
        csv_file = tmp_path / "test_data.csv"
        csv_file.write_text(
            "latitude,longitude,id,date\n"
            "43.0,-110.0,1,2024-01-01\n"
            "43.1,-110.1,2,2024-01-02\n"
        )
        return csv_file
    
    @pytest.fixture
    def sample_shapefile_data(self, tmp_path):
        """Create sample shapefile data."""
        # Create a simple GeoDataFrame and save as shapefile
        from shapely.geometry import Point, LineString
        
        points = [
            Point(-110.0, 43.0),
            Point(-110.1, 43.1),
            Point(-110.2, 43.2)
        ]
        
        gdf = gpd.GeoDataFrame(
            {'id': [1, 2, 3], 'route': ['A', 'A', 'A']},
            geometry=points,
            crs='EPSG:4326'
        )
        
        shapefile_dir = tmp_path / "shapefile"
        shapefile_dir.mkdir()
        shapefile_path = shapefile_dir / "test.shp"
        gdf.to_file(shapefile_path)
        
        return shapefile_path
    
    def test_process_csv(self, sample_csv_data, tmp_path):
        """Test CSV processing."""
        # Create a temporary copy of the real script to avoid modifying the real file
        import importlib.util
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "process_raw_presence_data.py"
        temp_script = tmp_path / "process_raw_presence_data.py"
        
        # Copy the real script to temp location
        if real_script.exists():
            shutil.copy2(real_script, temp_script)
        else:
            # If real script doesn't exist, skip this test
            pytest.skip("Real process_raw_presence_data.py not found")
        
        # Import from temporary location
        spec = importlib.util.spec_from_file_location(
            "process_raw_presence_data",
            temp_script
        )
        process_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(process_module)
        
        # Verify the function exists
        assert hasattr(process_module, 'process_csv'), "process_csv function not found in module"
        
        df = process_module.process_csv(sample_csv_data, "test_dataset")
        
        assert len(df) == 2
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
        assert all(df['latitude'].between(-90, 90))
        assert all(df['longitude'].between(-180, 180))
    
    def test_process_shapefile(self, sample_shapefile_data, tmp_path):
        """Test shapefile processing."""
        # Create a temporary copy of the real script to avoid modifying the real file
        import importlib.util
        import shutil
        scripts_dir = Path(__file__).parent.parent / "scripts"
        real_script = scripts_dir / "process_raw_presence_data.py"
        temp_script = tmp_path / "process_raw_presence_data.py"
        
        # Copy the real script to temp location
        if real_script.exists():
            shutil.copy2(real_script, temp_script)
        else:
            # If real script doesn't exist, skip this test
            pytest.skip("Real process_raw_presence_data.py not found")
        
        # Import from temporary location
        spec = importlib.util.spec_from_file_location(
            "process_raw_presence_data",
            temp_script
        )
        process_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(process_module)
        
        # Verify the function exists
        assert hasattr(process_module, 'process_shapefile'), "process_shapefile function not found in module"
        
        df = process_module.process_shapefile(sample_shapefile_data, "test_dataset")
        
        assert len(df) == 3
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
        assert all(df['latitude'].between(-90, 90))
        assert all(df['longitude'].between(-180, 180))


class TestEndToEndPipeline:
    """End-to-end integration tests with small test dataset."""
    
    @pytest.fixture
    def test_data_setup(self, tmp_path):
        """Set up a complete test data environment."""
        # Create directory structure
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"
        dem_dir = data_dir / "dem"
        terrain_dir = data_dir / "terrain"
        landcover_dir = data_dir / "landcover"
        canopy_dir = data_dir / "canopy"
        hydrology_dir = data_dir / "hydrology"
        infrastructure_dir = data_dir / "infrastructure"
        boundaries_dir = data_dir / "boundaries"
        
        for d in [raw_dir, processed_dir, dem_dir, terrain_dir, landcover_dir,
                  canopy_dir, hydrology_dir, infrastructure_dir, boundaries_dir]:
            d.mkdir(parents=True)
        
        # Create test raw data (CSV with a few points)
        test_dataset_dir = raw_dir / "elk_test_dataset"
        test_dataset_dir.mkdir()
        test_csv = test_dataset_dir / "test_data.csv"
        test_csv.write_text(
            "latitude,longitude,id,date\n"
            "43.0,-110.0,1,2024-01-01\n"
            "43.1,-110.1,2,2024-01-02\n"
            "43.2,-110.2,3,2024-01-03\n"
        )
        
        # Create minimal mock environmental data files
        # (These would normally be large rasters, but for testing we'll create minimal ones)
        import json
        
        # Create mock water sources GeoJSON
        water_data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-110.0, 43.0]},
                "properties": {"water_type": "stream"}
            }]
        }
        (hydrology_dir / "water_sources.geojson").write_text(json.dumps(water_data))
        
        # Create mock Wyoming boundary
        wyoming_bbox = {
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
                },
                "properties": {"name": "Wyoming"}
            }]
        }
        (boundaries_dir / "wyoming_state.shp").parent.mkdir(exist_ok=True)
        # For shapefile, we'd need to use geopandas, but for testing we'll skip this
        
        return {
            'data_dir': data_dir,
            'raw_dir': raw_dir,
            'processed_dir': processed_dir,
            'test_csv': test_csv
        }
    
    @patch('subprocess.run')
    def test_end_to_end_pipeline_with_mocks(self, mock_run, test_data_setup):
        """Test end-to-end pipeline execution with mocked subprocess calls."""
        # Mock successful subprocess runs
        mock_run.return_value = Mock(returncode=0)
        
        data_dir = test_data_setup['data_dir']
        
        # Create pipeline
        pipeline = DataPipeline(
            data_dir=data_dir,
            dataset_name='test_dataset',
            skip_steps=[],
            force=False
        )
        
        # Mock all script paths to exist
        scripts_dir = Path(__file__).parent.parent / "scripts"
        for step in pipeline.steps:
            # Use real script paths if they exist, otherwise create mocks
            if not step.script_path.exists():
                step.script_path.parent.mkdir(parents=True, exist_ok=True)
                step.script_path.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(0)")
                step.script_path.chmod(0o755)
        
        # Create required input files
        presence_file = test_data_setup['processed_dir'] / "test_dataset_points.csv"
        presence_file.parent.mkdir(parents=True, exist_ok=True)
        presence_file.write_text("latitude,longitude\n43.0,-110.0\n")
        
        combined_file = test_data_setup['processed_dir'] / "combined_test_dataset_presence_absence.csv"
        combined_file.write_text("latitude,longitude,elk_present\n43.0,-110.0,1\n43.1,-110.1,0\n")
        
        # Run pipeline (will use mocked subprocess)
        # Note: This test verifies the pipeline structure, not actual execution
        step_names = [step.name for step in pipeline.steps]
        assert len(step_names) > 0
        assert 'process_raw' in step_names
    
    def test_pipeline_handles_missing_inputs_gracefully(self, test_data_setup):
        """Test that pipeline handles missing inputs gracefully."""
        data_dir = test_data_setup['data_dir']
        
        # Create pipeline with dataset that doesn't exist
        pipeline = DataPipeline(
            data_dir=data_dir,
            dataset_name='nonexistent_dataset',
            skip_steps=[],
            force=False
        )
        
        # Check that steps requiring missing inputs report can_run() as False
        for step in pipeline.steps:
            if step.required_input and not step.required_input.exists():
                assert step.can_run() is False

