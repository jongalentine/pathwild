"""
Tests for temporal absence generators.

Tests the four enhanced temporal absence generation strategies to ensure they
produce valid absence points with complete temporal metadata.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

try:
    from src.data.temporal_absence_generators import (
        TemporallyMatchedAbsenceGenerator,
        SeasonalSegregationAbsenceGenerator,
        UnsuitableTemporalEnvironmentalAbsenceGenerator,
        RandomTemporalBackgroundGenerator,
        sample_date_from_distribution,
        get_season_from_month,
        days_in_month
    )
    TEMPORAL_GENERATORS_AVAILABLE = True
except ImportError:
    TEMPORAL_GENERATORS_AVAILABLE = False
    pytest.skip("Temporal absence generators not available", allow_module_level=True)


@pytest.fixture
def sample_presence_data_with_dates():
    """Create sample presence data with dates for testing."""
    np.random.seed(42)
    lats = np.random.uniform(43.0, 44.0, 20)
    lons = np.random.uniform(-107.5, -107.0, 20)
    
    # Create dates across different months/years
    dates = pd.to_datetime([
        '2020-06-15', '2020-07-20', '2020-08-10', '2020-09-05',
        '2020-10-12', '2020-11-18', '2020-12-25', '2021-01-08',
        '2021-02-14', '2021-03-22', '2021-04-30', '2021-05-15',
        '2021-06-20', '2021-07-25', '2021-08-12', '2021-09-18',
        '2021-10-22', '2021-11-28', '2021-12-10', '2022-01-15'
    ])
    
    # DatetimeIndex has .year, .month, .dayofyear directly (no .dt accessor needed)
    df = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'date': dates,
        'year': dates.year,  # DatetimeIndex has .year directly
        'month': dates.month,  # DatetimeIndex has .month directly
        'day_of_year': dates.dayofyear,  # DatetimeIndex has .dayofyear directly
        'elk_present': 1,
        'dataset': 'test_dataset'
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


class TestTemporalUtilityFunctions:
    """Test utility functions for temporal operations."""
    
    def test_get_season_from_month(self):
        """Test season mapping from month."""
        assert get_season_from_month(1) == 'wi'  # January
        assert get_season_from_month(6) == 'su'  # June
        assert get_season_from_month(9) == 'fa'  # September
        assert get_season_from_month(12) == 'wi'  # December
    
    def test_days_in_month(self):
        """Test days in month calculation."""
        assert days_in_month(1, 2020) == 31  # January
        assert days_in_month(2, 2020) == 29  # February (leap year)
        assert days_in_month(2, 2021) == 28  # February (non-leap year)
        assert days_in_month(4, 2020) == 30  # April
    
    def test_sample_date_from_distribution(self, sample_presence_data_with_dates):
        """Test date sampling from presence distribution."""
        presence_df = pd.DataFrame(sample_presence_data_with_dates.drop(columns='geometry'))
        
        # Sample multiple dates
        sampled_dates = [sample_date_from_distribution(presence_df) for _ in range(100)]
        
        # All should be valid dates
        assert all(isinstance(d, pd.Timestamp) for d in sampled_dates)
        
        # Should sample from presence date range
        presence_dates = pd.to_datetime(presence_df['date'])
        min_date = presence_dates.min()
        max_date = presence_dates.max()
        
        # Most sampled dates should be within presence range (allowing some variation)
        sampled_in_range = sum(1 for d in sampled_dates if min_date <= d <= max_date)
        assert sampled_in_range > 50, "Most dates should be within presence date range"


class TestTemporallyMatchedAbsenceGenerator:
    """Test TemporallyMatchedAbsenceGenerator."""
    
    def test_generate_with_temporal_metadata(self, sample_presence_data_with_dates, sample_study_area, temp_data_dir):
        """Test that generated absences have complete temporal metadata."""
        generator = TemporallyMatchedAbsenceGenerator(
            sample_presence_data_with_dates,
            sample_study_area,
            data_dir=temp_data_dir,
            min_distance_meters=2000.0
        )
        
        absences = generator.generate(n_samples=10, max_attempts=5000)
        
        assert len(absences) > 0
        assert 'latitude' in absences.columns
        assert 'longitude' in absences.columns
        assert 'date' in absences.columns
        assert 'year' in absences.columns
        assert 'month' in absences.columns
        assert 'day_of_year' in absences.columns
        assert 'season' in absences.columns
        assert 'absence_strategy' in absences.columns
        assert all(absences['absence_strategy'] == 'temporal_matched')
        assert all(absences['elk_present'] == 0)
        
        # Check temporal metadata completeness
        assert absences['date'].notna().all(), "All absences should have dates"
        assert absences['year'].notna().all(), "All absences should have year"
        assert absences['month'].notna().all(), "All absences should have month"
        assert absences['day_of_year'].notna().all(), "All absences should have day_of_year"
        
        # Check that months are valid (1-12)
        assert all(1 <= m <= 12 for m in absences['month'].dropna())
        
        # Check that years are reasonable
        assert all(2000 <= y <= 2025 for y in absences['year'].dropna())


class TestSeasonalSegregationAbsenceGenerator:
    """Test SeasonalSegregationAbsenceGenerator."""
    
    def test_generate_with_seasonal_offset(self, sample_presence_data_with_dates, sample_study_area):
        """Test that seasonal segregation uses same locations but offset dates."""
        generator = SeasonalSegregationAbsenceGenerator(
            sample_presence_data_with_dates,
            sample_study_area,
            date_column='date',
            offset_months=6
        )
        
        absences = generator.generate(n_samples=10)
        
        assert len(absences) > 0
        assert 'date' in absences.columns
        assert 'year' in absences.columns
        assert 'month' in absences.columns
        assert 'absence_strategy' in absences.columns
        assert all(absences['absence_strategy'] == 'seasonal_segregation')
        assert all(absences['confidence'] == 'high')
        
        # Check that dates are offset (approximately 6 months)
        # Note: Some variation is expected due to day-of-month handling
        if 'original_date' in absences.columns:
            for idx, row in absences.iterrows():
                original = pd.to_datetime(row['original_date'])
                new_date = pd.to_datetime(row['date'])
                # Should be approximately 6 months apart (allow ±1 month for day-of-month issues)
                month_diff = abs((new_date.year - original.year) * 12 + (new_date.month - original.month))
                assert 5 <= month_diff <= 7, f"Date offset should be ~6 months, got {month_diff}"
        
        # Check temporal metadata completeness
        assert absences['date'].notna().all()
        assert absences['year'].notna().all()
        assert absences['month'].notna().all()


class TestUnsuitableTemporalEnvironmentalAbsenceGenerator:
    """Test UnsuitableTemporalEnvironmentalAbsenceGenerator."""
    
    def test_generate_with_unsuitable_rules(self, sample_presence_data_with_dates, sample_study_area, temp_data_dir):
        """Test that unsuitable temporal-environmental absences are generated."""
        config = {
            'strategies': {
                'unsuitable_temporal_environmental': {
                    'rules': {
                        'winter_too_high': {
                            'elevation_min': 9000,
                            'months': [12, 1, 2, 3]
                        },
                        'summer_too_low': {
                            'elevation_max': 6000,
                            'months': [6, 7, 8]
                        }
                    }
                }
            }
        }
        
        generator = UnsuitableTemporalEnvironmentalAbsenceGenerator(
            sample_presence_data_with_dates,
            sample_study_area,
            data_dir=temp_data_dir,
            config=config
        )
        
        absences = generator.generate(n_samples=10, max_attempts=5000)
        
        if len(absences) > 0:
            assert 'date' in absences.columns
            assert 'year' in absences.columns
            assert 'month' in absences.columns
            assert 'absence_strategy' in absences.columns
            assert all(absences['absence_strategy'] == 'unsuitable_temporal_env')
            assert all(absences['confidence'] == 'high')
            
            # Check that months match rule months
            months = absences['month'].dropna().unique()
            rule_months = {12, 1, 2, 3, 6, 7, 8}
            assert all(m in rule_months for m in months), \
                f"All months should be in rule months, got {months}"


class TestRandomTemporalBackgroundGenerator:
    """Test RandomTemporalBackgroundGenerator."""
    
    def test_generate_with_uniform_temporal(self, sample_presence_data_with_dates, sample_study_area):
        """Test that random temporal background generates uniform temporal distribution."""
        generator = RandomTemporalBackgroundGenerator(
            sample_presence_data_with_dates,
            sample_study_area,
            min_distance_meters=500.0
        )
        
        absences = generator.generate(n_samples=50, max_attempts=5000)
        
        assert len(absences) > 0
        assert 'date' in absences.columns
        assert 'year' in absences.columns
        assert 'month' in absences.columns
        assert 'absence_strategy' in absences.columns
        assert all(absences['absence_strategy'] == 'random_background')
        assert all(absences['confidence'] == 'low')
        
        # Check temporal metadata completeness
        assert absences['date'].notna().all()
        assert absences['year'].notna().all()
        assert absences['month'].notna().all()
        
        # Check that months are distributed (should have multiple months)
        unique_months = absences['month'].dropna().nunique()
        assert unique_months > 1, "Should have multiple months in uniform distribution"


class TestTemporalMetadataCompleteness:
    """Test that all temporal generators produce complete temporal metadata."""
    
    @pytest.mark.parametrize("generator_class,generator_kwargs", [
        (TemporallyMatchedAbsenceGenerator, {'min_distance_meters': 2000.0}),
        (RandomTemporalBackgroundGenerator, {'min_distance_meters': 500.0}),
    ])
    def test_temporal_metadata_completeness(
        self,
        generator_class,
        generator_kwargs,
        sample_presence_data_with_dates,
        sample_study_area,
        temp_data_dir
    ):
        """Test that all temporal generators produce complete temporal metadata."""
        if generator_class == TemporallyMatchedAbsenceGenerator:
            generator = generator_class(
                sample_presence_data_with_dates,
                sample_study_area,
                data_dir=temp_data_dir,
                **generator_kwargs
            )
        else:
            generator = generator_class(
                sample_presence_data_with_dates,
                sample_study_area,
                **generator_kwargs
            )
        
        absences = generator.generate(n_samples=10, max_attempts=5000)
        
        if len(absences) > 0:
            # All absences should have temporal metadata
            temporal_cols = ['date', 'year', 'month', 'day_of_year']
            for col in temporal_cols:
                if col in absences.columns:
                    missing = absences[col].isna().sum()
                    assert missing == 0, f"{col} should not have missing values, got {missing} missing"
            
            # Check that temporal values are valid
            assert all(1 <= m <= 12 for m in absences['month'].dropna())
            assert all(2000 <= y <= 2025 for y in absences['year'].dropna())
            assert all(1 <= d <= 366 for d in absences['day_of_year'].dropna())

