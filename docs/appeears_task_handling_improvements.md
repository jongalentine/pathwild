# AppEEARS Task Handling Improvements

## Overview

Enhanced the AppEEARS client to better handle failed, stuck, and timeout scenarios. The pipeline now gracefully continues processing even when some tasks fail or timeout.

## Improvements Implemented

### 1. Enhanced Status Detection

**Before:** Only detected `"done"` and `"failed"` statuses.

**After:** Now detects multiple terminal states:
- `"done"` - Successfully completed
- `"failed"` - Explicitly failed
- `"error"` - Error state
- `"cancelled"` - Cancelled by user or system
- `"invalid"` - Invalid request

All failed states are now properly detected and handled, preventing infinite polling on stuck tasks.

### 2. Improved Logging

**Failed Tasks:**
- Logs task ID (truncated for readability)
- Logs number of points affected
- Logs error message from AppEEARS API
- Example: `Task 8f069e81... (3 points): Task cancelled by user`

**Timeout Tasks:**
- Logs which tasks timed out
- Shows number of points affected
- For small batches (≤5 points), shows individual point coordinates and dates
- Example:
  ```
  Timeout: 1 task(s) did not complete within 30 minutes
    - Task 8f069e81... (3 point(s)): Timed out after 30 minutes
      Point: (41.1395, -109.7018) on 2007-11-17
      Point: (43.5803, -110.6687) on 2013-02-07
      Point: (44.1982, -110.3998) on 2014-07-13
  ```

**Summary Logging:**
- Comprehensive summary at end of polling
- Shows completed, failed, and timeout counts
- Example:
  ```
  Completed polling: 7/8 tasks finished successfully
  Failed tasks: 0/8
  Timeout: 1 task(s) did not complete within 30 minutes
  ```

### 3. Graceful Failure Handling

**Before:** Failed tasks would cause the pipeline to wait the full timeout period.

**After:**
- Failed tasks are immediately detected and removed from polling
- Only successfully completed tasks have their results downloaded
- Failed/stuck tasks are logged but don't block processing
- Points from failed tasks remain as placeholders (can be retried later)

### 4. Better Result Tracking

**Summary Statistics:**
- Logs how many results came from API vs cache
- Shows how many points are missing (will remain as placeholders)
- Example:
  ```
  NDVI retrieval summary: 10 from API, 2 from cache, 1 missing (will remain as placeholders)
  ```

## Behavior Changes

### Pipeline Continuation

The pipeline **will continue processing** even if some tasks fail or timeout:

1. **Failed Tasks:** Detected immediately, logged, skipped
2. **Timeout Tasks:** After 30 minutes, logged, skipped
3. **Completed Tasks:** Results downloaded and processed normally
4. **Missing Points:** Remain as placeholders in feature files

### No Pipeline Failure

The pipeline will **not fail** due to:
- Individual task failures
- Task timeouts
- Network errors during polling
- API rate limiting (handled with retries)

The pipeline will only fail if:
- All tasks fail (no results at all)
- Critical errors in result processing
- Authentication failures

## Example Scenarios

### Scenario 1: One Task Fails

```
Submitted 8 tasks...
Polling iteration 5: 7 pending, 1 completed
Task abc12345 failed: Invalid coordinates
Completed polling: 7/8 tasks finished successfully
Failed tasks: 1/8
  - Task abc12345... (3 points): Invalid coordinates
Skipping 1 failed/stuck task(s) affecting 3 point(s)
Downloading results for 7 successfully completed tasks...
NDVI retrieval summary: 20 from API, 2 from cache, 3 missing (will remain as placeholders)
```

### Scenario 2: One Task Times Out

```
Submitted 8 tasks...
Polling iteration 120: 1 pending, 7 completed (elapsed: 20m 0s)
Timeout: 1 task(s) did not complete within 30 minutes
  - Task def67890... (2 point(s)): Timed out after 30 minutes
    Point: (41.1395, -109.7018) on 2007-11-17
    Point: (43.5803, -110.6687) on 2013-02-07
Completed polling: 7/8 tasks finished successfully
Skipping 1 failed/stuck task(s) affecting 2 point(s)
Downloading results for 7 successfully completed tasks...
NDVI retrieval summary: 21 from API, 1 from cache, 2 missing (will remain as placeholders)
```

## Benefits

1. **Resilience:** Pipeline continues even with partial failures
2. **Visibility:** Clear logging shows exactly what failed and why
3. **Efficiency:** Failed tasks don't block successful ones
4. **Debugging:** Detailed point-level information for troubleshooting
5. **User Experience:** Pipeline completes successfully with partial results

## Future Enhancements (Optional)

Potential future improvements:
1. **Automatic Retry:** Retry failed tasks with exponential backoff
2. **Task Status Dashboard:** Real-time status of all tasks
3. **Partial Result Caching:** Cache partial results from failed tasks
4. **Notification System:** Alert when tasks fail or timeout
5. **Retry Queue:** Queue failed points for later retry
