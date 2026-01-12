"""
Lunar Illumination Calculator

Calculates lunar illumination metrics for wildlife activity modeling.
Based on ecological research showing that ungulate movement patterns
are influenced by nocturnal light conditions.

Key features:
    - moon_phase: Fraction of moon illuminated (0-1)
    - moon_altitude: Moon's elevation above horizon (degrees)
    - effective_illumination: Combined metric accounting for moon position
    - cloud_adjusted_illumination: Effective illumination reduced by cloud cover

Research background:
    - Ungulates show "lunar phobic" or "lunar philic" behavior depending on species
    - Elk tend to be more active on darker nights (lunar phobic)
    - Cloud cover significantly reduces effective moonlight
    - Moon altitude affects ground-level illumination

References:
    - Lazzeri et al. (2021) "Carried away by a moonlight shadow" - Mammal Research
    - Penteriani et al. (2023) "Biologically meaningful moonlight measures" - Behav Ecol Sociobiol
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try to import ephem, provide fallback if not available
try:
    import ephem
    EPHEM_AVAILABLE = True
except ImportError:
    EPHEM_AVAILABLE = False
    logger.warning("ephem not installed. Install with: pip install ephem")


class LunarCalculator:
    """
    Calculator for lunar illumination metrics.

    Provides multiple metrics for modeling wildlife response to moonlight:
    - Basic moon phase (simple, commonly used)
    - Moon altitude (position-aware)
    - Effective illumination (phase + altitude combined)
    - Cloud-adjusted illumination (accounts for weather)
    """

    # Wyoming approximate center (for default calculations)
    DEFAULT_LAT = 43.0
    DEFAULT_LON = -108.0

    def __init__(self):
        """Initialize lunar calculator."""
        if not EPHEM_AVAILABLE:
            logger.warning("LunarCalculator initialized without ephem - will use fallback calculations")

    def get_moon_phase(self, date: Union[datetime, str]) -> float:
        """
        Get moon phase as illuminated fraction (0-1).

        Args:
            date: Date/datetime or 'YYYY-MM-DD' string

        Returns:
            Moon phase as fraction (0 = new moon, 1 = full moon)
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        if EPHEM_AVAILABLE:
            moon = ephem.Moon(date)
            return moon.phase / 100.0  # ephem returns 0-100
        else:
            # Fallback: simple lunar cycle approximation
            # Synodic month = 29.53 days
            return self._fallback_moon_phase(date)

    def _fallback_moon_phase(self, date: datetime) -> float:
        """
        Fallback moon phase calculation without ephem.

        Uses a known new moon date and the synodic month period.
        """
        # Known new moon: January 11, 2024
        known_new_moon = datetime(2024, 1, 11)
        synodic_month = 29.53059  # days

        days_since = (date - known_new_moon).total_seconds() / 86400
        phase_angle = (days_since % synodic_month) / synodic_month * 2 * math.pi

        # Illumination follows a cosine curve
        # 0 at new moon, 1 at full moon
        illumination = (1 - math.cos(phase_angle)) / 2
        return illumination

    def get_moon_position(
        self,
        lat: float,
        lon: float,
        date: Union[datetime, str],
        hour: int = 0
    ) -> Dict[str, float]:
        """
        Get moon position (altitude and azimuth) for a location and time.

        Args:
            lat: Latitude
            lon: Longitude
            date: Date or datetime
            hour: Hour of day (0-23, default midnight for nocturnal activity)

        Returns:
            Dictionary with:
                - altitude: Degrees above horizon (-90 to 90)
                - azimuth: Degrees from north (0-360)
                - is_visible: True if moon is above horizon
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        # Set time
        dt = datetime(date.year, date.month, date.day, hour, 0, 0)

        if EPHEM_AVAILABLE:
            observer = ephem.Observer()
            observer.lat = str(lat)
            observer.lon = str(lon)
            observer.date = dt

            moon = ephem.Moon(observer)

            altitude_deg = math.degrees(float(moon.alt))
            azimuth_deg = math.degrees(float(moon.az))

            return {
                'altitude': altitude_deg,
                'azimuth': azimuth_deg,
                'is_visible': altitude_deg > 0
            }
        else:
            # Fallback: return approximate values
            # This is a very rough approximation
            phase = self._fallback_moon_phase(date)
            # Moon is roughly opposite to sun, higher at night during full moon
            approx_altitude = 45 * phase - 10  # Very rough estimate

            return {
                'altitude': approx_altitude,
                'azimuth': 180.0,  # Approximate south
                'is_visible': approx_altitude > 0
            }

    def get_effective_illumination(
        self,
        lat: float,
        lon: float,
        date: Union[datetime, str],
        hour: int = 0
    ) -> float:
        """
        Calculate effective lunar illumination at ground level.

        Combines moon phase and altitude to estimate actual light
        reaching the ground. Based on research showing that moon
        altitude significantly affects ground-level illumination.

        Args:
            lat: Latitude
            lon: Longitude
            date: Date or datetime
            hour: Hour of day (default: midnight)

        Returns:
            Effective illumination (0-1 scale)
            0 = no moonlight, 1 = maximum possible moonlight
        """
        phase = self.get_moon_phase(date)
        position = self.get_moon_position(lat, lon, date, hour)

        altitude = position['altitude']

        # If moon is below horizon, no illumination
        if altitude <= 0:
            return 0.0

        # Altitude factor: illumination increases with altitude
        # Maximum effect around 30-45 degrees altitude
        # Uses sine function - peaks at 90 degrees
        altitude_rad = math.radians(min(altitude, 90))
        altitude_factor = math.sin(altitude_rad)

        # Atmospheric extinction: light is reduced more at low angles
        # Air mass approximation (simplified Kasten-Young formula)
        if altitude > 0:
            air_mass = 1 / (math.sin(altitude_rad) + 0.50572 * (altitude + 6.07995) ** -1.6364)
            air_mass = min(air_mass, 10)  # Cap at horizon
            extinction_factor = math.exp(-0.1 * air_mass)  # Simple extinction model
        else:
            extinction_factor = 0

        # Combine factors
        effective = phase * altitude_factor * extinction_factor

        return min(1.0, max(0.0, effective))

    def get_cloud_adjusted_illumination(
        self,
        lat: float,
        lon: float,
        date: Union[datetime, str],
        cloud_cover_pct: float,
        hour: int = 0
    ) -> float:
        """
        Calculate lunar illumination adjusted for cloud cover.

        This is the key metric for wildlife modeling - it represents
        the actual light conditions animals experience at night.

        Args:
            lat: Latitude
            lon: Longitude
            date: Date or datetime
            cloud_cover_pct: Cloud cover percentage (0-100)
            hour: Hour of day (default: midnight)

        Returns:
            Cloud-adjusted illumination (0-1 scale)
        """
        effective = self.get_effective_illumination(lat, lon, date, hour)

        # Cloud transmission factor
        # Thick clouds block most moonlight, thin clouds less so
        # Using exponential model: transmission = exp(-k * cloud_cover)
        cloud_fraction = cloud_cover_pct / 100.0

        # k factor determines how much clouds block light
        # Higher k = more blocking. k=2 means 50% clouds block ~63% of light
        k = 2.5
        transmission = math.exp(-k * cloud_fraction)

        return effective * transmission

    def get_all_lunar_features(
        self,
        lat: float,
        lon: float,
        date: Union[datetime, str],
        cloud_cover_pct: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Get all lunar features for a location and date.

        Calculates features at multiple times of night to capture
        the full nocturnal illumination pattern.

        Args:
            lat: Latitude
            lon: Longitude
            date: Date or datetime
            cloud_cover_pct: Cloud cover percentage (0-100), optional

        Returns:
            Dictionary with all lunar features:
                - moon_phase: Illuminated fraction (0-1)
                - moon_altitude_midnight: Altitude at midnight
                - moon_altitude_mean: Mean altitude during night hours
                - moon_visible_hours: Hours moon is above horizon
                - effective_illumination_midnight: At midnight
                - effective_illumination_mean: Mean during night
                - cloud_adjusted_illumination: If cloud cover provided
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        features = {}

        # Basic moon phase
        features['moon_phase'] = self.get_moon_phase(date)

        # Calculate for night hours (8 PM to 6 AM = hours 20-23, 0-5)
        night_hours = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]

        altitudes = []
        illuminations = []
        visible_hours = 0

        for hour in night_hours:
            # Adjust date for hours after midnight
            check_date = date if hour >= 12 else date + timedelta(days=1)

            position = self.get_moon_position(lat, lon, check_date, hour)
            altitudes.append(position['altitude'])

            if position['is_visible']:
                visible_hours += 1

            illum = self.get_effective_illumination(lat, lon, check_date, hour)
            illuminations.append(illum)

        # Midnight values (hour 0)
        midnight_idx = night_hours.index(0)
        features['moon_altitude_midnight'] = altitudes[midnight_idx]
        features['effective_illumination_midnight'] = illuminations[midnight_idx]

        # Mean values across night
        features['moon_altitude_mean'] = sum(altitudes) / len(altitudes)
        features['effective_illumination_mean'] = sum(illuminations) / len(illuminations)
        features['moon_visible_hours'] = visible_hours

        # Cloud-adjusted illumination (if cloud cover provided)
        if cloud_cover_pct is not None:
            features['cloud_adjusted_illumination'] = self.get_cloud_adjusted_illumination(
                lat, lon, date, cloud_cover_pct, hour=0
            )
            # Also provide mean cloud-adjusted illumination
            cloud_adjusted = []
            for hour in night_hours:
                check_date = date if hour >= 12 else date + timedelta(days=1)
                adj = self.get_cloud_adjusted_illumination(lat, lon, check_date, cloud_cover_pct, hour)
                cloud_adjusted.append(adj)
            features['cloud_adjusted_illumination_mean'] = sum(cloud_adjusted) / len(cloud_adjusted)

        return features

    def get_lunar_features_batch(
        self,
        locations: list,
        cloud_cover_data: Optional[Dict] = None
    ) -> Dict[tuple, Dict[str, float]]:
        """
        Calculate lunar features for multiple locations efficiently.

        Args:
            locations: List of (lat, lon, date) tuples
            cloud_cover_data: Optional dict mapping (lat, lon, date) to cloud cover %

        Returns:
            Dictionary mapping (lat, lon, date) to feature dictionaries
        """
        results = {}

        for lat, lon, date in locations:
            if isinstance(date, datetime):
                date_key = date.strftime('%Y-%m-%d')
            else:
                date_key = date

            key = (lat, lon, date_key)

            # Get cloud cover if available
            cloud_cover = None
            if cloud_cover_data is not None:
                cloud_cover = cloud_cover_data.get(key)

            results[key] = self.get_all_lunar_features(lat, lon, date, cloud_cover)

        return results


# Convenience function for simple usage
def get_lunar_illumination(
    lat: float,
    lon: float,
    date: Union[datetime, str],
    cloud_cover_pct: Optional[float] = None
) -> float:
    """
    Get effective lunar illumination for a location and date.

    Convenience function for simple usage. For batch processing,
    use LunarCalculator class directly.

    Args:
        lat: Latitude
        lon: Longitude
        date: Date or datetime or 'YYYY-MM-DD' string
        cloud_cover_pct: Optional cloud cover percentage (0-100)

    Returns:
        Effective illumination (0-1), adjusted for clouds if provided
    """
    calc = LunarCalculator()

    if cloud_cover_pct is not None:
        return calc.get_cloud_adjusted_illumination(lat, lon, date, cloud_cover_pct)
    else:
        return calc.get_effective_illumination(lat, lon, date)
