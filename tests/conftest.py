"""
Shared pytest fixtures for AppEEARS batching tests.

Provides test data fixtures and mock API response fixtures.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import json


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def points_same_date():
    """Points with identical dates - should batch into one request."""
    return [
        (44.0, -107.0, "2024-06-15"),
        (44.1, -107.1, "2024-06-15"),
        (44.2, -107.2, "2024-06-15"),
        (44.3, -107.3, "2024-06-15"),
        (44.4, -107.4, "2024-06-15"),
    ]


@pytest.fixture
def points_different_dates():
    """Points with different dates - may need multiple batches depending on buffer."""
    return [
        (44.0, -107.0, "2024-06-10"),
        (44.1, -107.1, "2024-06-20"),
        (44.2, -107.2, "2024-06-30"),
        (44.3, -107.3, "2024-07-10"),
        (44.4, -107.4, "2024-07-20"),
    ]


@pytest.fixture
def points_mixed_dates():
    """Points with mixed dates - some same, some different."""
    return [
        (44.0, -107.0, "2024-06-15"),
        (44.1, -107.1, "2024-06-15"),
        (44.2, -107.2, "2024-06-20"),
        (44.3, -107.3, "2024-06-20"),
        (44.4, -107.4, "2024-07-01"),
        (44.5, -107.5, "2024-07-01"),
        (44.6, -107.6, "2024-07-15"),
    ]


@pytest.fixture
def points_large_batch():
    """Large batch of points for testing batch size limits."""
    base_date = datetime(2024, 6, 15)
    points = []
    for i in range(150):
        lat = 44.0 + (i * 0.01)
        lon = -107.0 - (i * 0.01)
        # Vary dates slightly
        date = (base_date + timedelta(days=i % 7)).strftime("%Y-%m-%d")
        points.append((lat, lon, date))
    return points


@pytest.fixture
def points_wyoming_region():
    """Realistic points from Wyoming region (where PathWild operates)."""
    return [
        # Northern Bighorn Mountains
        (44.85687, -106.48514, "2024-06-15"),
        (44.87215, -106.51093, "2024-06-15"),
        (44.89085, -106.51986, "2024-06-15"),
        # Southern Bighorn Mountains
        (44.37215, -104.81093, "2024-06-20"),
        (44.39085, -104.71986, "2024-06-20"),
        # National Elk Refuge
        (43.58033, -110.66872, "2024-06-25"),
        (43.54693, -110.64062, "2024-06-25"),
        # Southern GYE
        (44.19822, -110.39982, "2024-07-01"),
        (44.22154, -110.38458, "2024-07-01"),
    ]


# ============================================================================
# Mock API Response Fixtures
# ============================================================================

@pytest.fixture
def mock_appeears_auth_response():
    """Mock AppEEARS authentication response."""
    return {
        "token": "test_bearer_token_12345",
        "token_type": "Bearer",
        "expiration": "2024-12-31T23:59:59Z"
    }


@pytest.fixture
def mock_appeears_task_submission_response():
    """Mock AppEEARS task submission response."""
    return {
        "task_id": "test_task_batch_abc123",
        "status": "pending"
    }


@pytest.fixture
def mock_appeears_task_status_pending():
    """Mock AppEEARS task status (pending)."""
    return {
        "task_id": "test_task_batch_abc123",
        "status": "pending",
        "created": "2024-06-15T10:00:00Z",
        "updated": "2024-06-15T10:00:00Z",
        "user_id": "test_user@example.com",
        "task_name": "pathwild_ndvi_20240615_100000",
        "task_type": "point"
    }


@pytest.fixture
def mock_appeears_task_status_done():
    """Mock AppEEARS task status (done)."""
    return {
        "task_id": "test_task_batch_abc123",
        "status": "done",
        "created": "2024-06-15T10:00:00Z",
        "updated": "2024-06-15T10:05:00Z",
        "completed": "2024-06-15T10:05:00Z",
        "user_id": "test_user@example.com",
        "task_name": "pathwild_ndvi_20240615_100000",
        "task_type": "point",
        "attempts": 1
    }


@pytest.fixture
def mock_appeears_bundle_files():
    """Mock AppEEARS bundle file listing."""
    return {
        "files": [
            {
                "file_id": "file_result_123",
                "file_name": "pathwild_ndvi_20240615_100000-MOD13Q1_061-results.csv",
                "file_size": 2048,
                "file_type": "csv",
                "sha256": "abc123def456..."
            },
            {
                "file_id": "file_metadata_456",
                "file_name": "pathwild_ndvi_20240615_100000-MOD13Q1_061-metadata.xml",
                "file_size": 512,
                "file_type": "xml",
                "sha256": "def456ghi789..."
            }
        ],
        "task_id": "test_task_batch_abc123",
        "created": "2024-06-15T10:05:00Z",
        "updated": "2024-06-15T10:05:00Z",
        "bundle_type": "point"
    }


@pytest.fixture
def mock_appeears_csv_content_batched():
    """Mock AppEEARS CSV content with multiple coordinates."""
    return """Date,Latitude,Longitude,MOD13Q1_061__250m_16_days_NDVI,MOD13Q1_061__250m_16_days_VI_Quality
2024-06-10,44.0,-107.0,6500,0
2024-06-15,44.0,-107.0,7000,0
2024-06-20,44.0,-107.0,6800,0
2024-06-10,44.1,-107.1,6200,0
2024-06-15,44.1,-107.1,6700,0
2024-06-20,44.1,-107.1,6500,0
2024-06-10,44.2,-107.2,7000,0
2024-06-15,44.2,-107.2,7200,0
2024-06-20,44.2,-107.2,7100,0
2024-06-10,44.3,-107.3,6800,0
2024-06-15,44.3,-107.3,7100,0
2024-06-20,44.3,-107.3,6900,0
2024-06-10,44.4,-107.4,6400,0
2024-06-15,44.4,-107.4,6900,0
2024-06-20,44.4,-107.4,6700,0"""


@pytest.fixture
def mock_appeears_csv_content_single():
    """Mock AppEEARS CSV content for single coordinate."""
    return """Date,Latitude,Longitude,MOD13Q1_061__250m_16_days_NDVI,MOD13Q1_061__250m_16_days_VI_Quality
2024-06-10,44.0,-107.0,6500,0
2024-06-15,44.0,-107.0,7000,0
2024-06-20,44.0,-107.0,6800,0"""


@pytest.fixture
def mock_appeears_product_layers():
    """Mock AppEEARS product layers response."""
    return {
        "_250m_16_days_NDVI": {
            "Description": "16 day NDVI average",
            "DataType": "int16",
            "FillValue": -3000,
            "ValidMin": -2000,
            "ValidMax": 10000,
            "ScaleFactor": 0.0001,
            "Units": "NDVI"
        },
        "_250m_16_days_EVI": {
            "Description": "16 day EVI average",
            "DataType": "int16",
            "FillValue": -3000,
            "ValidMin": -2000,
            "ValidMax": 10000,
            "ScaleFactor": 0.0001,
            "Units": "EVI"
        }
    }


# ============================================================================
# Combined Fixtures
# ============================================================================

@pytest.fixture
def mock_appeears_responses(
    mock_appeears_auth_response,
    mock_appeears_task_submission_response,
    mock_appeears_task_status_pending,
    mock_appeears_task_status_done,
    mock_appeears_bundle_files
):
    """Combined fixture with all mock AppEEARS responses."""
    return {
        "auth": mock_appeears_auth_response,
        "task_submission": mock_appeears_task_submission_response,
        "task_status_pending": mock_appeears_task_status_pending,
        "task_status_done": mock_appeears_task_status_done,
        "bundle_files": mock_appeears_bundle_files
    }
