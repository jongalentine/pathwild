# PRISM Data Persistence Strategy

## Overview

PRISM climate data files are **permanently persisted** once downloaded. This is critical for performance and cost efficiency, especially for historical training data which never changes.

## Key Improvements

### 1. Rate Limit Tracking

**Problem**: When PRISM rate limit is hit (2 downloads per file per 24 hours), the file wasn't saved, causing repeated failed attempts.

**Solution**: 
- Created rate limit marker files (`.ratelimit`) that track when we hit rate limits
- Markers prevent repeated download attempts for 24 hours
- Markers automatically expire after 23 hours (1 hour buffer)

**Files**: `data/prism/{variable}/prism_{variable}_us_4km_{date}.ratelimit`

### 2. Improved File Caching

**Problem**: ZIP files were downloaded but not always extracted, causing repeated extraction attempts.

**Solution**:
- `_get_file_path()` now automatically extracts COG files from existing ZIP files
- Checks file size to ensure files are valid
- Better logging for cache hits vs misses

### 3. Persistent Storage

**Strategy**: Once a PRISM file is downloaded, it is **never re-downloaded** unless:
- The file is manually deleted
- The file is corrupted (detected by size checks)

**Storage Location**: `data/prism/{variable}/prism_{variable}_us_4km_{date}.tif`

**Benefits**:
- Historical training data (2017-2023) downloaded once, reused forever
- Faster pipeline runs (no repeated downloads)
- Respects PRISM rate limits
- Reduces network usage

## File Formats

### COG Format (Current - Web Service)
- **Source**: PRISM web service (https://services.nacse.org/prism/data/get/)
- **Format**: Cloud Optimized GeoTIFF (.tif)
- **Delivery**: ZIP file containing .tif and ancillary files
- **Storage**: Extracted .tif file is kept, ZIP is deleted after extraction

### BIL Format (Legacy - FTP)
- **Source**: Old FTP service (deprecated Sept 2025)
- **Format**: Band Interleaved by Line (.bil)
- **Storage**: .bil file kept for backward compatibility

## Rate Limit Handling

PRISM allows **2 downloads per file per 24 hours**. Our implementation:

1. **First attempt**: Downloads file normally
2. **Second attempt** (within 24h): Downloads successfully
3. **Third+ attempts** (within 24h): Rate limit hit
   - Creates `.ratelimit` marker file
   - Returns `None` (no file available)
   - Skips all future attempts for 24 hours
4. **After 24h**: Marker expires, attempts resume

## Cache Management

### Automatic Extraction
If a ZIP file exists but COG is missing, the system automatically extracts it on first access.

### Manual Cleanup
To force re-download of a file:
```bash
# Remove specific file
rm data/prism/ppt/prism_ppt_us_4km_20180513.tif

# Remove rate limit marker (if stuck)
rm data/prism/ppt/prism_ppt_us_4km_20180513.ratelimit

# Remove all files for a date
rm data/prism/*/prism_*_us_4km_20180513.*
```

## Performance Impact

### Before
- Repeated download attempts for rate-limited files
- ~1.6-1.7 rows/sec processing speed
- Many failed download attempts in logs

### After
- Rate limit markers prevent wasted attempts
- Files persist across pipeline runs
- Faster processing (fewer network calls)
- Better logging (cache hits vs misses)

## Best Practices

1. **Pre-download Historical Data**: For large training datasets, consider pre-downloading all required PRISM files before running the pipeline
2. **Monitor Cache**: Check `data/prism/` directory size - files are ~50-100MB each
3. **Respect Rate Limits**: Don't manually delete and re-download files frequently
4. **Use Git LFS**: Consider Git LFS for version control of PRISM files (if needed)

## Future Enhancements

1. **Batch Download Script**: Pre-download all required dates for a dataset
2. **Cache Validation**: Verify file integrity (checksums)
3. **Compression**: Compress old files to save disk space
4. **S3 Backup**: Backup PRISM cache to S3 for team sharing

