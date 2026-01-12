"""
ERA5 Cloud Cover Client

Provides access to ERA5 reanalysis cloud cover data downloaded from
the ECMWF Climate Data Store. Data is stored as NetCDF files covering
the Wyoming region.

ERA5 is used for historical cloud cover because:
1. Bulk download: entire region/date range in one request
2. No rate limits: data is read from local files
3. Consistent quality: reanalysis data is gap-filled and quality-controlled
4. Appropriate resolution: 0.25° (~25km) is suitable for cloud cover

OpenMeteo should only be used for forecast/current data at inference time.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union
import threading

import numpy as np

logger = logging.getLogger(__name__)


class ERA5CloudClient:
    """
    Client for reading ERA5 cloud cover data from local NetCDF files.

    The data should be downloaded using scripts/bulk_download_era5_cloud.py
    which creates yearly NetCDF files covering Wyoming.

    Thread-safe: uses lazy loading with locking for xarray datasets.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize ERA5 cloud cover client.

        Args:
            data_dir: Directory containing ERA5 NetCDF files.
                      Defaults to data/era5/
        """
        if data_dir is None:
            data_dir = Path('data/era5')
        self.data_dir = Path(data_dir)

        # Lazy-loaded datasets (one per year)
        self._datasets: Dict[int, any] = {}
        self._lock = threading.Lock()

        # Track available years
        self._available_years: Optional[set] = None

        logger.info(f"ERA5 cloud client initialized: {self.data_dir}")

    def _get_available_years(self) -> set:
        """Get set of years with downloaded data."""
        if self._available_years is None:
            self._available_years = set()
            if self.data_dir.exists():
                for f in self.data_dir.glob('era5_cloud_cover_wyoming_*.nc'):
                    try:
                        year = int(f.stem.split('_')[-1])
                        self._available_years.add(year)
                    except ValueError:
                        pass
        return self._available_years

    def _load_dataset(self, year: int):
        """
        Load xarray dataset for a specific year.

        Uses lazy loading - datasets are only loaded when first accessed.
        """
        if year in self._datasets:
            return self._datasets[year]

        with self._lock:
            # Double-check after acquiring lock
            if year in self._datasets:
                return self._datasets[year]

            file_path = self.data_dir / f'era5_cloud_cover_wyoming_{year}.nc'

            if not file_path.exists():
                logger.warning(f"ERA5 data not found for {year}: {file_path}")
                self._datasets[year] = None
                return None

            try:
                import xarray as xr
                ds = xr.open_dataset(file_path)
                self._datasets[year] = ds
                logger.debug(f"Loaded ERA5 dataset for {year}")
                return ds
            except Exception as e:
                logger.error(f"Error loading ERA5 dataset for {year}: {e}")
                self._datasets[year] = None
                return None

    def get_cloud_cover(
        self,
        lat: float,
        lon: float,
        date: Union[datetime, str],
        time_of_day: str = 'daily_mean'
    ) -> Optional[float]:
        """
        Get cloud cover percentage for a location and date.

        Args:
            lat: Latitude
            lon: Longitude
            date: Date (datetime or 'YYYY-MM-DD' string)
            time_of_day: One of:
                - 'daily_mean': Average of all 4 time steps (default)
                - 'night': Average of 00:00 and 06:00 UTC (~6PM-midnight MT)
                - 'day': Average of 12:00 and 18:00 UTC (~6AM-noon MT)
                - '00:00', '06:00', '12:00', '18:00': Specific time

        Returns:
            Cloud cover as percentage (0-100), or None if not available
        """
        # Parse date
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        year = date.year

        # Check if year is available
        if year not in self._get_available_years():
            logger.debug(f"ERA5 data not available for year {year}")
            return None

        # Load dataset
        ds = self._load_dataset(year)
        if ds is None:
            return None

        try:
            # Select nearest grid point
            # ERA5 uses longitude in 0-360 format, but our data is -180 to 180
            # Check the actual coordinate range in the dataset
            if 'longitude' in ds.coords:
                lon_coord = 'longitude'
                lat_coord = 'latitude'
            elif 'lon' in ds.coords:
                lon_coord = 'lon'
                lat_coord = 'lat'
            else:
                logger.error("Unknown coordinate names in ERA5 dataset")
                return None

            # Handle longitude format
            ds_lon_min = float(ds[lon_coord].min())
            if ds_lon_min >= 0 and lon < 0:
                # Dataset uses 0-360, convert our -180 to 180 longitude
                lon = lon + 360

            # Select the date
            date_str = date.strftime('%Y-%m-%d')

            # Get the variable (could be 'tcc' or 'total_cloud_cover')
            if 'tcc' in ds:
                var_name = 'tcc'
            elif 'total_cloud_cover' in ds:
                var_name = 'total_cloud_cover'
            else:
                # Try to find any cloud-related variable
                for var in ds.data_vars:
                    if 'cloud' in var.lower():
                        var_name = var
                        break
                else:
                    logger.error(f"No cloud cover variable found in ERA5 dataset")
                    return None

            # Select data for the date and location
            data = ds[var_name].sel(
                **{lat_coord: lat, lon_coord: lon},
                method='nearest'
            ).sel(time=date_str)

            # Handle time_of_day selection
            if time_of_day == 'daily_mean':
                value = float(data.mean())
            elif time_of_day == 'night':
                # Night hours: 00:00 and 06:00 UTC
                night_data = data.sel(time=data.time.dt.hour.isin([0, 6]))
                value = float(night_data.mean()) if len(night_data) > 0 else float(data.mean())
            elif time_of_day == 'day':
                # Day hours: 12:00 and 18:00 UTC
                day_data = data.sel(time=data.time.dt.hour.isin([12, 18]))
                value = float(day_data.mean()) if len(day_data) > 0 else float(data.mean())
            else:
                # Specific time
                try:
                    hour = int(time_of_day.split(':')[0])
                    time_data = data.sel(time=data.time.dt.hour == hour)
                    value = float(time_data.mean()) if len(time_data) > 0 else float(data.mean())
                except:
                    value = float(data.mean())

            # ERA5 cloud cover is 0-1 fraction, convert to percentage
            if value <= 1.0:
                value = value * 100.0

            return value

        except KeyError as e:
            logger.debug(f"Date {date_str} not found in ERA5 data: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error reading ERA5 cloud cover for {lat}, {lon}, {date}: {e}")
            return None

    def get_cloud_cover_batch(
        self,
        locations: list,
        time_of_day: str = 'daily_mean'
    ) -> Dict[tuple, Optional[float]]:
        """
        Get cloud cover for multiple locations efficiently.

        Args:
            locations: List of (lat, lon, date) tuples
            time_of_day: Time selection (see get_cloud_cover)

        Returns:
            Dictionary mapping (lat, lon, date) to cloud cover percentage
        """
        results = {}

        # Group by year for efficient dataset access
        by_year: Dict[int, list] = {}
        for lat, lon, date in locations:
            if isinstance(date, str):
                date = datetime.strptime(date, '%Y-%m-%d')
            year = date.year
            if year not in by_year:
                by_year[year] = []
            by_year[year].append((lat, lon, date))

        # Process each year's locations
        for year, year_locations in by_year.items():
            for lat, lon, date in year_locations:
                key = (lat, lon, date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date)
                results[key] = self.get_cloud_cover(lat, lon, date, time_of_day)

        return results

    def is_available(self, year: int) -> bool:
        """Check if ERA5 data is available for a specific year."""
        return year in self._get_available_years()

    def get_coverage_info(self) -> Dict:
        """Get information about available ERA5 data."""
        available = self._get_available_years()

        if not available:
            return {
                'available': False,
                'years': [],
                'message': 'No ERA5 data downloaded. Run scripts/bulk_download_era5_cloud.py'
            }

        return {
            'available': True,
            'years': sorted(available),
            'year_range': (min(available), max(available)),
            'data_dir': str(self.data_dir)
        }

    def close(self):
        """Close all open datasets."""
        with self._lock:
            for ds in self._datasets.values():
                if ds is not None:
                    try:
                        ds.close()
                    except:
                        pass
            self._datasets.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
