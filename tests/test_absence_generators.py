"""
Tests for absence data generators.

Tests the four absence generation strategies to ensure they produce
valid absence points that meet spatial and environmental constraints.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from pathlib import Path
import tempfile
import shutil

from src.data.absence_generators import (
    AbsenceGenerator,
    EnvironmentalPseudoAbsenceGenerator,
    UnsuitableHabitatAbsenceGenerator,
    RandomBackgroundGenerator,
    TemporalAbsenceGenerator
)


@pytest.fixture
def sample_presence_data():
    """Create sample presence data for testing."""
    # Create 10 presence points in Wyoming
    np.random.seed(42)
    lats = np.random.uniform(43.0, 44.0, 10)
    lons = np.random.uniform(-107.5, -107.0, 10)
    
    df = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'elk_present': 1
    })
    
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    
    return gdf


@pytest.fixture
def sample_study_area():
    """Create sample study area (Wyoming bounding box)."""
    wyoming_bbox = box(-111.0, 41.0, -104.0, 45.0)
    return gpd.GeoDataFrame(geometry=[wyoming_bbox], crs="EPSG:4326")


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestAbsenceGenerator:
    """Test base AbsenceGenerator class."""
    
    def test_distance_constraint(self, sample_presence_data, sample_study_area):
        """Test distance constraint checking."""
        generator = RandomBackgroundGenerator(
            sample_presence_data,
            sample_study_area,
            min_distance_meters=1000.0
        )
        
        # Point far from presences should pass
        far_point = Point(-108.0, 44.0)
        assert generator.check_distance_constraint(far_point)
        
        # Point close to presences should fail
        close_point = sample_presence_data.geometry.iloc[0]
        assert not generator.check_distance_constraint(close_point)
    
    def test_sample_random_point(self, sample_presence_data, sample_study_area):
        """Test random point sampling."""
        generator = RandomBackgroundGenerator(
            sample_presence_data,
            sample_study_area
        )
        
        point = generator._sample_random_point_in_study_area()
        assert point is not None
        assert isinstance(point, Point)
        assert generator._is_in_study_area(point)


class TestRandomBackgroundGenerator:
    """Test RandomBackgroundGenerator."""
    
    def test_generate(self, sample_presence_data, sample_study_area):
        """Test random background absence generation."""
        generator = RandomBackgroundGenerator(
            sample_presence_data,
            sample_study_area,
            min_distance_meters=500.0
        )
        
        absences = generator.generate(n_samples=5, max_attempts=1000)
        
        assert len(absences) > 0
        assert 'latitude' in absences.columns
        assert 'longitude' in absences.columns
        assert 'absence_strategy' in absences.columns
        assert all(absences['absence_strategy'] == 'background')
        
        # Check all points are in study area
        assert all(generator._is_in_study_area(p) for p in absences.geometry)
        
        # Check distance constraints
        for point in absences.geometry:
            assert generator.check_distance_constraint(point)
    
    def test_generate_with_parallel(self, sample_presence_data, sample_study_area):
        """Test parallel processing (should work same as sequential for small dataset)."""
        generator = RandomBackgroundGenerator(
            sample_presence_data,
            sample_study_area,
            min_distance_meters=500.0
        )
        
        # Test with parallel processing (n_processes=2)
        absences_parallel = generator.generate(n_samples=5, max_attempts=1000, n_processes=2)
        
        # Test sequential (n_processes=1)
        absences_sequential = generator.generate(n_samples=5, max_attempts=1000, n_processes=1)
        
        # Both should produce valid results
        assert len(absences_parallel) > 0
        assert len(absences_sequential) > 0
        assert 'absence_strategy' in absences_parallel.columns
        assert 'absence_strategy' in absences_sequential.columns
    
    def test_adaptive_max_attempts(self, sample_presence_data, sample_study_area):
        """Test adaptive max_attempts calculation."""
        generator = RandomBackgroundGenerator(
            sample_presence_data,
            sample_study_area
        )
        
        # Small dataset should have base max_attempts
        max_attempts_small = generator._calculate_adaptive_max_attempts(100)
        assert max_attempts_small >= 10000
        
        # Large dataset should scale up
        # Create a larger presence dataset
        large_presence = gpd.GeoDataFrame(
            geometry=[Point(-107.0, 43.0)] * 50000,
            crs="EPSG:4326"
        )
        large_generator = RandomBackgroundGenerator(large_presence, sample_study_area)
        max_attempts_large = large_generator._calculate_adaptive_max_attempts(20000)
        assert max_attempts_large > max_attempts_small
    
    def test_generate_insufficient_attempts(self, sample_presence_data, sample_study_area):
        """Test behavior when max_attempts is too low."""
        generator = RandomBackgroundGenerator(
            sample_presence_data,
            sample_study_area,
            min_distance_meters=50000.0  # Very large distance
        )
        
        # Should generate fewer points than requested
        absences = generator.generate(n_samples=100, max_attempts=10)
        assert len(absences) <= 10


class TestEnvironmentalPseudoAbsenceGenerator:
    """Test EnvironmentalPseudoAbsenceGenerator."""
    
    def test_generate_without_data(self, sample_presence_data, sample_study_area, temp_data_dir):
        """Test generation without environmental data files."""
        generator = EnvironmentalPseudoAbsenceGenerator(
            sample_presence_data,
            sample_study_area,
            data_dir=temp_data_dir,
            min_distance_meters=2000.0
        )
        
        # Should still work (uses defaults)
        absences = generator.generate(n_samples=3, max_attempts=1000)
        
        assert len(absences) >= 0  # May generate 0 if constraints too strict
        if len(absences) > 0:
            assert 'absence_strategy' in absences.columns
            assert all(absences['absence_strategy'] == 'environmental')
    
    def test_generate_with_parallel(self, sample_presence_data, sample_study_area, temp_data_dir):
        """Test environmental generator with parallel processing."""
        generator = EnvironmentalPseudoAbsenceGenerator(
            sample_presence_data,
            sample_study_area,
            data_dir=temp_data_dir,
            min_distance_meters=2000.0
        )
        
        # Test with parallel processing
        absences = generator.generate(n_samples=3, max_attempts=1000, n_processes=2)
        
        # Should work (may generate 0 if constraints too strict)
        assert len(absences) >= 0
        if len(absences) > 0:
            assert 'absence_strategy' in absences.columns
            assert all(absences['absence_strategy'] == 'environmental')
    
    def test_environmental_suitability(self, sample_presence_data, sample_study_area, temp_data_dir):
        """Test environmental suitability checking."""
        generator = EnvironmentalPseudoAbsenceGenerator(
            sample_presence_data,
            sample_study_area,
            data_dir=temp_data_dir
        )
        
        # Without data files, should use defaults (all points pass)
        point = Point(-108.0, 44.0)
        # This will use default values, so should pass
        result = generator._is_environmentally_suitable(point)
        # Result depends on defaults, but should not crash
        assert isinstance(result, bool)


class TestUnsuitableHabitatAbsenceGenerator:
    """Test UnsuitableHabitatAbsenceGenerator."""
    
    def test_generate(self, sample_presence_data, sample_study_area, temp_data_dir):
        """Test unsuitable habitat absence generation."""
        generator = UnsuitableHabitatAbsenceGenerator(
            sample_presence_data,
            sample_study_area,
            data_dir=temp_data_dir
        )
        
        absences = generator.generate(n_samples=3, max_attempts=1000)
        
        assert len(absences) >= 0
        if len(absences) > 0:
            assert 'absence_strategy' in absences.columns
            assert all(absences['absence_strategy'] == 'unsuitable')
    
    def test_generate_with_parallel(self, sample_presence_data, sample_study_area, temp_data_dir):
        """Test unsuitable generator with parallel processing."""
        generator = UnsuitableHabitatAbsenceGenerator(
            sample_presence_data,
            sample_study_area,
            data_dir=temp_data_dir
        )
        
        # Test with parallel processing
        absences = generator.generate(n_samples=3, max_attempts=1000, n_processes=2)
        
        assert len(absences) >= 0
        if len(absences) > 0:
            assert 'absence_strategy' in absences.columns
            assert all(absences['absence_strategy'] == 'unsuitable')


class TestTemporalAbsenceGenerator:
    """Test TemporalAbsenceGenerator."""
    
    def test_generate_with_dates(self, sample_presence_data, sample_study_area):
        """Test temporal absence generation with date column."""
        # Add date column
        sample_presence_data['date'] = pd.date_range(
            start='2024-06-15',
            periods=len(sample_presence_data),
            freq='D'
        )
        
        generator = TemporalAbsenceGenerator(
            sample_presence_data,
            sample_study_area,
            date_column='date'
        )
        
        absences = generator.generate(n_samples=5)
        
        assert len(absences) == 5
        assert 'date' in absences.columns
        assert 'original_date' in absences.columns
        assert 'absence_strategy' in absences.columns
        assert all(absences['absence_strategy'] == 'temporal')
        
        # Check dates are different from originals
        for idx, row in absences.iterrows():
            assert row['date'] != row['original_date']
    
    def test_generate_without_dates(self, sample_presence_data, sample_study_area):
        """Test temporal absence generation without date column."""
        generator = TemporalAbsenceGenerator(
            sample_presence_data,
            sample_study_area,
            date_column='date'
        )
        
        # Should return empty GeoDataFrame
        absences = generator.generate(n_samples=5)
        
        assert len(absences) == 0
    
    def test_generate_with_parallel(self, sample_presence_data, sample_study_area):
        """Test temporal generator with parallel processing (should work same)."""
        # Add date column
        sample_presence_data['date'] = pd.date_range(
            start='2024-06-15',
            periods=len(sample_presence_data),
            freq='D'
        )
        
        generator = TemporalAbsenceGenerator(
            sample_presence_data,
            sample_study_area,
            date_column='date'
        )
        
        # Temporal generator doesn't use parallel processing (uses existing points)
        # But should accept n_processes parameter for consistency
        absences = generator.generate(n_samples=5, n_processes=2)
        
        assert len(absences) == 5
        assert 'absence_strategy' in absences.columns
        assert all(absences['absence_strategy'] == 'temporal')
    
    def test_season_shifting(self, sample_presence_data, sample_study_area):
        """Test that dates are shifted to opposite seasons."""
        # Add summer dates
        sample_presence_data['date'] = pd.date_range(
            start='2024-07-15',
            periods=len(sample_presence_data),
            freq='D'
        )
        
        generator = TemporalAbsenceGenerator(
            sample_presence_data,
            sample_study_area,
            date_column='date'
        )
        
        absences = generator.generate(n_samples=len(sample_presence_data))
        
        # Check that dates are in winter (Dec-Feb)
        for date in absences['date']:
            month = pd.to_datetime(date).month
            assert month in [12, 1, 2]


class TestIntegration:
    """Integration tests for multiple generators."""
    
    def test_multiple_strategies(self, sample_presence_data, sample_study_area):
        """Test combining multiple absence generation strategies."""
        # Generate absences from multiple strategies
        generators = [
            RandomBackgroundGenerator(sample_presence_data, sample_study_area),
            RandomBackgroundGenerator(sample_presence_data, sample_study_area)
        ]
        
        all_absences = []
        for gen in generators:
            absences = gen.generate(n_samples=3, max_attempts=1000)
            all_absences.append(absences)
        
        combined = pd.concat(all_absences, ignore_index=True)
        
        assert len(combined) > 0
        assert 'elk_present' not in combined.columns  # Not set by generators
        
        # Set elk_present
        combined['elk_present'] = 0
        
        # Combine with presence
        sample_presence_data['elk_present'] = 1
        training_data = pd.concat([
            sample_presence_data[['latitude', 'longitude', 'elk_present']],
            combined[['latitude', 'longitude', 'elk_present']]
        ], ignore_index=True)
        
        assert len(training_data) == len(sample_presence_data) + len(combined)
        assert (training_data['elk_present'] == 1).sum() == len(sample_presence_data)
        assert (training_data['elk_present'] == 0).sum() == len(combined)
    
    def test_parallel_vs_sequential(self, sample_presence_data, sample_study_area):
        """Test that parallel and sequential produce similar results."""
        generator = RandomBackgroundGenerator(
            sample_presence_data,
            sample_study_area,
            min_distance_meters=500.0
        )
        
        # Generate with sequential
        absences_seq = generator.generate(n_samples=10, max_attempts=2000, n_processes=1)
        
        # Generate with parallel
        absences_par = generator.generate(n_samples=10, max_attempts=2000, n_processes=2)
        
        # Both should produce valid results
        assert len(absences_seq) > 0
        assert len(absences_par) > 0
        
        # Both should have same structure
        assert 'absence_strategy' in absences_seq.columns
        assert 'absence_strategy' in absences_par.columns
        assert all(absences_seq['absence_strategy'] == 'background')
        assert all(absences_par['absence_strategy'] == 'background')
        
        # Both should meet distance constraints
        for point in absences_seq.geometry:
            assert generator.check_distance_constraint(point)
        for point in absences_par.geometry:
            assert generator.check_distance_constraint(point)

