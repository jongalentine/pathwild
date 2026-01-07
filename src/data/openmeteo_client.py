"""
Open-Meteo Client for Weather Forecasts

Handles weather forecast data retrieval for inference pipeline.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


class OpenMeteoClient:
    """Client for Open-Meteo weather forecast API"""
    
    FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_BASE = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, forecast_days: int = 7, timeout: int = 30):
        """
        Initialize Open-Meteo client.
        
        Args:
            forecast_days: Number of forecast days to request (default: 7)
            timeout: Request timeout in seconds
        """
        self.forecast_days = forecast_days
        self.timeout = timeout
        self.cache = {}  # Simple in-memory cache
    
    def _make_request(
        self,
        base_url: str,
        params: Dict,
        max_retries: int = 3
    ) -> Dict:
        """
        Make API request with retry logic.
        
        Args:
            base_url: API base URL
            params: Request parameters
            max_retries: Maximum retry attempts
            
        Returns:
            JSON response as dictionary
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    base_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"API request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to get Open-Meteo data after {max_retries} attempts: {e}")
    
    def get_forecast(
        self,
        lat: float,
        lon: float,
        timezone: str = "America/Denver"
    ) -> Dict:
        """
        Get weather forecast for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            timezone: Timezone (default: America/Denver for Wyoming)
            
        Returns:
            Dictionary with forecast data
        """
        # Check cache
        cache_key = f"forecast_{lat:.4f}_{lon:.4f}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            # Cache for 1 hour
            if (datetime.now() - cached_time).seconds < 3600:
                return cached_data
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "forecast_days": self.forecast_days,
            "timezone": timezone
        }
        
        data = self._make_request(self.FORECAST_BASE, params)
        
        # Cache result
        self.cache[cache_key] = (data, datetime.now())
        
        return data
    
    def get_historical(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        timezone: str = "America/Denver"
    ) -> Dict:
        """
        Get historical weather data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timezone: Timezone
            
        Returns:
            Dictionary with historical data
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": timezone
        }
        
        return self._make_request(self.HISTORICAL_BASE, params)
    
    def parse_forecast_response(self, data: Dict) -> List[Dict]:
        """
        Parse Open-Meteo forecast response into list of daily records.
        
        Args:
            data: Raw API response
            
        Returns:
            List of dictionaries with daily forecast data
        """
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        temp_mean = daily.get("temperature_2m_mean", [])
        precipitation = daily.get("precipitation_sum", [])
        
        results = []
        for i, date_str in enumerate(dates):
            result = {
                "date": date_str,
                "temp_max_f": temp_max[i] if i < len(temp_max) else None,
                "temp_min_f": temp_min[i] if i < len(temp_min) else None,
                "temp_mean_f": temp_mean[i] if i < len(temp_mean) else None,
                "precipitation_inches": precipitation[i] if i < len(precipitation) else None
            }
            results.append(result)
        
        return results
    
    def get_forecast_for_location(
        self,
        lat: float,
        lon: float,
        timezone: str = "America/Denver"
    ) -> List[Dict]:
        """
        Get parsed forecast data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            timezone: Timezone
            
        Returns:
            List of daily forecast dictionaries
        """
        data = self.get_forecast(lat, lon, timezone)
        return self.parse_forecast_response(data)
    
    def get_forecast_for_date(
        self,
        lat: float,
        lon: float,
        target_date: str,
        timezone: str = "America/Denver"
    ) -> Optional[Dict]:
        """
        Get forecast for a specific future date.
        
        Args:
            lat: Latitude
            lon: Longitude
            target_date: Target date (YYYY-MM-DD)
            timezone: Timezone
            
        Returns:
            Forecast dictionary for the target date, or None if not in forecast range
        """
        forecasts = self.get_forecast_for_location(lat, lon, timezone)
        
        for forecast in forecasts:
            if forecast["date"] == target_date:
                return forecast
        
        return None

