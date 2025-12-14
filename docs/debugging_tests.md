# Debugging Pytest Tests

## Quick Start

### 1. Drop into debugger on test failure
```bash
# Single test
pytest tests/test_heuristics.py::TestElevationHeuristic::test_optimal_elevation_october --pdb

# All tests, stop on first failure
pytest tests/test_heuristics.py --pdb -x

# Drop into debugger before test runs
pytest tests/test_heuristics.py::TestElevationHeuristic::test_optimal_elevation_october --trace
```

### 2. Add breakpoint() in your code
Add `breakpoint()` anywhere in your test or source code:

```python
def test_optimal_elevation_october(self):
    """Test that optimal October elevation (8500-9500) scores 10"""
    heuristic = ElevationHeuristic()
    
    location = {"lat": 43.0, "lon": -110.0}
    date = "2026-10-15"
    context = {"elevation": 9000.0}
    
    breakpoint()  # Execution stops here
    
    result = heuristic.calculate(location, date, context)
    
    assert result.score == 10.0
```

### 3. Use IDE Debugger (Cursor/VS Code)

1. **Set breakpoints**: Click in the gutter (left of line numbers) to set breakpoints
2. **Run in debug mode**: 
   - Press `F5` or go to Run > Start Debugging
   - Select "Python: Pytest" configuration
   - Or use the test icon next to test functions

3. **Create launch.json** (if needed):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "tests/",
                "-v",
                "--tb=short"
            ],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

## Debugger Commands (pdb/ipdb)

Once in the debugger, use these commands:

- `n` (next) - Execute next line
- `s` (step) - Step into function calls
- `c` (continue) - Continue execution
- `l` (list) - Show current code context
- `p <variable>` - Print variable value
- `pp <variable>` - Pretty print variable
- `u` (up) - Move up stack frame
- `d` (down) - Move down stack frame
- `w` (where) - Show stack trace
- `q` (quit) - Quit debugger
- `h` (help) - Show help

## Better Debugging with ipdb

Install ipdb for a better debugging experience:
```bash
pip install ipdb
```

Then use `import ipdb; ipdb.set_trace()` instead of `breakpoint()`:
```python
import ipdb; ipdb.set_trace()  # Better than breakpoint()
```

Or use pytest-ipdb plugin:
```bash
pip install pytest-ipdb
pytest --ipdb  # Drops into ipdb on failures
```

## Common Debugging Scenarios

### Debug a specific failing test
```bash
pytest tests/test_heuristics.py::TestElevationHeuristic::test_optimal_elevation_october --pdb -v
```

### Debug with verbose output
```bash
pytest tests/test_heuristics.py -vv --pdb
```

### Debug and see print statements
```bash
pytest tests/test_heuristics.py -s --pdb
```

### Debug multiple tests matching a pattern
```bash
pytest tests/test_heuristics.py -k "elevation" --pdb
```

## Tips

1. **Use `-s` flag** to see print statements: `pytest -s`
2. **Use `-vv` flag** for very verbose output
3. **Use `-k` flag** to filter tests: `pytest -k "elevation"`
4. **Set breakpoints in source code** to debug implementation issues
5. **Use `pytest --lf`** to re-run only the last failed tests

## Example: Debugging a failing test

```bash
# 1. Run test to see failure
pytest tests/test_heuristics.py::TestElevationHeuristic::test_optimal_elevation_october -v

# 2. Add breakpoint in test
# Edit test file, add: breakpoint()

# 3. Run with debugger
pytest tests/test_heuristics.py::TestElevationHeuristic::test_optimal_elevation_october -s

# 4. When breakpoint hits, inspect variables:
# (Pdb) p result
# (Pdb) p result.score
# (Pdb) p context
# (Pdb) n  # Step to next line
```
