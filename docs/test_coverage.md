# Running Tests with Coverage

## Quick Commands

### Basic Coverage Report
```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_heuristics.py --cov=src --cov-report=term-missing

# Run specific test
pytest tests/test_heuristics.py::TestElevationHeuristic::test_optimal_elevation_october --cov=src --cov-report=term-missing
```

### Coverage Report Options

```bash
# Terminal report (shows missing lines)
pytest --cov=src --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=src --cov-report=html
# Then open: htmlcov/index.html

# Both terminal and HTML
pytest --cov=src --cov-report=term-missing --cov-report=html

# XML report (for CI/CD)
pytest --cov=src --cov-report=xml

# JSON report
pytest --cov=src --cov-report=json
```

### Coverage Thresholds

```bash
# Fail if coverage is below threshold
pytest --cov=src --cov-report=term-missing --cov-fail-under=80

# Common thresholds:
# --cov-fail-under=70  # 70% minimum
# --cov-fail-under=80  # 80% minimum
# --cov-fail-under=90  # 90% minimum
```

### Excluding Files from Coverage

```bash
# Exclude specific paths
pytest --cov=src --cov-report=term-missing --cov-config=.coveragerc

# Or use inline exclusion
pytest --cov=src --cov-report=term-missing --ignore-glob='*/tests/*' --ignore-glob='*/__pycache__/*'
```

## Recommended Workflow

### Daily Development
```bash
# Quick coverage check
pytest --cov=src --cov-report=term-missing -v

# Focus on specific module
pytest tests/test_heuristics.py --cov=src.scoring.heuristics --cov-report=term-missing
```

### Before Committing
```bash
# Full coverage with HTML report
pytest --cov=src --cov-report=term-missing --cov-report=html

# Check coverage percentage
pytest --cov=src --cov-report=term --cov-fail-under=70
```

### CI/CD Pipeline
```bash
# XML report for coverage services (Codecov, Coveralls, etc.)
pytest --cov=src --cov-report=xml --cov-report=term
```

## Understanding Coverage Reports

### Terminal Output Example
```
Name                                    Stmts   Miss  Cover   Missing
--------------------------------------------------------------------
src/scoring/heuristics/elevation.py       45      2    96%   23-24
src/scoring/aggregator.py                 78     12    85%   45-50, 67-72
--------------------------------------------------------------------
TOTAL                                    123     14    89%
```

- **Stmts**: Total statements
- **Miss**: Missed statements
- **Cover**: Coverage percentage
- **Missing**: Line numbers not covered

### HTML Report
The HTML report provides:
- Line-by-line coverage highlighting
- File browser
- Branch coverage
- Interactive exploration

## Configuration File (Optional)

Create `.coveragerc` in project root for persistent settings:

```ini
[run]
source = src
omit = 
    */tests/*
    */__pycache__/*
    */venv/*
    */env/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

Then run:
```bash
pytest --cov --cov-report=term-missing
```

## Common Use Cases

### 1. Check coverage for new code
```bash
pytest tests/test_new_feature.py --cov=src.new_feature --cov-report=term-missing
```

### 2. Find untested code
```bash
pytest --cov=src --cov-report=term-missing | grep -A 5 "Missing"
```

### 3. Coverage for specific package
```bash
pytest --cov=src.scoring --cov-report=term-missing
```

### 4. Compare coverage before/after
```bash
# Before changes
pytest --cov=src --cov-report=json -o coverage_before.json

# After changes
pytest --cov=src --cov-report=json -o coverage_after.json

# Compare (requires coverage tool)
coverage combine coverage_before.json coverage_after.json
coverage report
```

## Tips

1. **Aim for 80%+ coverage** for production code
2. **Focus on critical paths** - don't obsess over 100%
3. **Use HTML reports** to visually see what's missing
4. **Exclude test files** from coverage calculations
5. **Set coverage thresholds** in CI/CD to enforce standards

## Integration with pytest.ini

You can add coverage to your pytest.ini:

```ini
[pytest]
addopts = 
    -v
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html
```

Then just run: `pytest` (coverage runs automatically)
