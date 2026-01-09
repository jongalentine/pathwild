# Changelog

## [Unreleased] - 2026-01-03

### Added
- **Retry Logic with Exponential Backoff**: AWDB API calls now automatically retry on 5xx server errors with exponential backoff (1s, 2s, 4s delays, max 3 retries)
- **Warning Suppression**: Class-level, thread-safe warning tracking to prevent duplicate log messages in parallel processing scenarios
- **Comprehensive Test Coverage**: Added `TestRetryLogic` and `TestWarningSuppression` test classes with 8 new tests covering retry behavior and warning suppression
- **Progress Reporting Consistency**: Standardized progress reporting format across all pipeline steps using tqdm with consistent batch completion messages
- **Debug Level Logging**: Cache loading and worker start messages moved to debug level to reduce log noise

### Changed
- **AWDB API Error Handling**: Improved error handling with safe status code checking for mock compatibility in tests
- **Pyproj Transform Calls**: Fixed deprecation warnings by ensuring scalar float values are passed to transformer.transform()
- **Logging Levels**: Cache loading messages changed from INFO to DEBUG level

### Fixed
- **Pyproj Deprecation Warnings**: Fixed "Conversion of an array with ndim > 0 to a scalar" warnings by explicitly converting to float
- **Test Mock Compatibility**: Fixed test failures by using proper exception types (`requests.exceptions.RequestException` instead of generic `Exception`)
- **Warning Suppression in Tests**: Added autouse fixture to reset warning sets between tests to prevent test pollution

### Technical Details

#### Retry Logic
- Location: `src/data/processors.py` - `AWDBClient._fetch_station_data_from_awdb()`
- Retries only 5xx server errors (500, 502, 503, etc.)
- Non-5xx errors fail immediately without retry
- Uses `getattr()` and `isinstance()` for safe status code checking

#### Warning Suppression
- Class-level sets: `AWDBClient._warned_stations` and `AWDBClient._warned_api_failures`
- Thread-safe using `threading.Lock()`
- Tracks warnings by (station_id, warning_type) tuples

#### Files Modified
- `src/data/processors.py` - Retry logic, warning suppression, pyproj fixes
- `src/data/absence_generators.py` - Debug level logging, pyproj fixes
- `scripts/integrate_environmental_features.py` - Progress reporting consistency
- `tests/test_snotel_integration.py` - New test classes for retry and warning suppression
- `docs/awdb_api_research.md` - Implementation status documentation
- `docs/automated_data_pipeline.md` - Updated feature list

