#!/usr/bin/env python3
"""
Estimate AppEEARS batch counts based on feature files.

Analyzes feature files to determine:
1. How many points have placeholder NDVI values
2. Date distribution of those points
3. Estimated number of batches needed (based on date_buffer_days=5 and max_points_per_batch=100)
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

# Placeholder values
NDVI_PLACEHOLDERS = {0.3, 0.5, 0.55, 0.7}
SUMMER_NDVI_PLACEHOLDER = 60.0

def reconstruct_date_from_sin_cos(year: float, month: float, day_of_year_sin: float, day_of_year_cos: float) -> Optional[datetime]:
    """
    Reconstruct date from year, month, and day_of_year_sin/cos.
    
    The sin/cos encoding is: sin(2π * day_of_year / 365.25) and cos(2π * day_of_year / 365.25)
    To reverse: day_of_year = atan2(sin, cos) * 365.25 / (2π)
    """
    if pd.isna(year) or pd.isna(month):
        return None
    
    try:
        year_int = int(year)
        month_int = int(month)
        
        if not (1 <= month_int <= 12):
            return None
        
        # Reconstruct day_of_year from sin/cos
        if not (pd.isna(day_of_year_sin) or pd.isna(day_of_year_cos)):
            # Use atan2 to get angle, then convert to day_of_year
            angle = math.atan2(day_of_year_sin, day_of_year_cos)
            # Normalize angle to [0, 2π)
            if angle < 0:
                angle += 2 * math.pi
            # Convert back to day_of_year
            day_of_year = (angle / (2 * math.pi)) * 365.25
            day_of_year = int(round(day_of_year))
            
            # Convert day_of_year to month/day
            base_date = datetime(year_int, 1, 1)
            target_date = base_date + timedelta(days=day_of_year - 1)
            return target_date
        else:
            # Fallback: use month and assume mid-month
            return datetime(year_int, month_int, 15)
    except (ValueError, TypeError, OverflowError):
        return None

# Batching parameters
DATE_BUFFER_DAYS = 5  # Default from appeears_client.py
MAX_POINTS_PER_BATCH = 100  # Default from appeears_client.py

def calculate_date_range_width(date_buffer_days):
    """Calculate the width of date range for grouping."""
    # Each point has: target_date ± date_buffer_days
    # So range width = 2 * date_buffer_days + 1 (the target date itself)
    return 2 * date_buffer_days + 1

def estimate_batches_for_points(points_with_dates, date_buffer_days=5, max_points_per_batch=100):
    """
    Estimate number of batches needed for a set of points.
    
    This simulates the grouping logic from _group_points_by_date_range.
    """
    if not points_with_dates:
        return 0
    
    # Calculate date ranges for each point
    point_ranges = []
    for date_str in points_with_dates:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = target_date - timedelta(days=date_buffer_days)
        end_date = target_date + timedelta(days=date_buffer_days)
        point_ranges.append((start_date, end_date, date_str))
    
    # Group points with overlapping date ranges
    groups = []
    used = set()
    
    for i, (start1, end1, date1) in enumerate(point_ranges):
        if i in used:
            continue
        
        group = [date1]
        used.add(i)
        
        # Find other points with overlapping date ranges
        for j, (start2, end2, date2) in enumerate(point_ranges[i+1:], start=i+1):
            if j in used:
                continue
            
            # Check if date ranges overlap
            if start1 <= end2 and start2 <= end1:
                # Check if adding this point would exceed batch size
                if len(group) < max_points_per_batch:
                    group.append(date2)
                    used.add(j)
        
        groups.append(group)
    
    return len(groups)

def analyze_feature_file(file_path):
    """Analyze a feature file for placeholder NDVI values."""
    path = Path(file_path)
    if not path.exists():
        return None
    
    df = pd.read_csv(path)
    dataset_name = path.stem.replace('_features', '')
    
    # Count placeholder NDVI values
    placeholder_ndvi_count = 0
    placeholder_summer_count = 0
    
    if 'ndvi' in df.columns:
        placeholder_ndvi_count = df['ndvi'].isin(NDVI_PLACEHOLDERS).sum()
    
    if 'summer_integrated_ndvi' in df.columns:
        placeholder_summer_count = (df['summer_integrated_ndvi'] == SUMMER_NDVI_PLACEHOLDER).sum()
    
    # Get points that need NDVI replacement (have placeholders)
    needs_replacement = []
    replacement_dates_list = []
    
    # Create mask for rows needing replacement
    needs_ndvi_replacement = pd.Series([False] * len(df))
    needs_summer_replacement = pd.Series([False] * len(df))
    
    if 'ndvi' in df.columns:
        needs_ndvi_replacement = df['ndvi'].isin(NDVI_PLACEHOLDERS)
    
    if 'summer_integrated_ndvi' in df.columns:
        needs_summer_replacement = (df['summer_integrated_ndvi'] == SUMMER_NDVI_PLACEHOLDER)
    
    # Combine masks (OR logic - need replacement if either is placeholder)
    needs_replacement_mask = needs_ndvi_replacement | needs_summer_replacement
    
    # Extract dates for points needing replacement
    if needs_replacement_mask.any() and all(col in df.columns for col in ['year', 'month']):
        replacement_df = df[needs_replacement_mask].copy()
        
        for _, row in replacement_df.iterrows():
            year = row.get('year')
            month = row.get('month')
            day_of_year_sin = row.get('day_of_year_sin', np.nan)
            day_of_year_cos = row.get('day_of_year_cos', np.nan)
            
            date_obj = reconstruct_date_from_sin_cos(year, month, day_of_year_sin, day_of_year_cos)
            if date_obj:
                date_str = date_obj.strftime('%Y-%m-%d')
                replacement_dates_list.append(date_str)
                needs_replacement.append([date_str, row.get('latitude'), row.get('longitude')])
    
    # Reconstruct dates from year, month, day_of_year_sin, day_of_year_cos
    reconstructed_dates = []
    if all(col in df.columns for col in ['year', 'month']):
        for _, row in df.iterrows():
            year = row.get('year')
            month = row.get('month')
            day_of_year_sin = row.get('day_of_year_sin', np.nan)
            day_of_year_cos = row.get('day_of_year_cos', np.nan)
            
            date_obj = reconstruct_date_from_sin_cos(year, month, day_of_year_sin, day_of_year_cos)
            if date_obj:
                reconstructed_dates.append(date_obj)
    
    # Get date range
    date_range_info = {}
    if reconstructed_dates:
        date_range_info = {
            'min_date': min(reconstructed_dates),
            'max_date': max(reconstructed_dates),
            'span_days': (max(reconstructed_dates) - min(reconstructed_dates)).days,
            'unique_dates': len(set(d.date() for d in reconstructed_dates))
        }
    
    # Get date range for points needing replacement
    replacement_date_range = {}
    replacement_dates_objs = []
    if needs_replacement and 'year' in df.columns and 'month' in df.columns:
        replacement_df = df[needs_replacement_mask].copy()
        for _, row in replacement_df.iterrows():
            year = row.get('year')
            month = row.get('month')
            day_of_year_sin = row.get('day_of_year_sin', np.nan)
            day_of_year_cos = row.get('day_of_year_cos', np.nan)
            
            date_obj = reconstruct_date_from_sin_cos(year, month, day_of_year_sin, day_of_year_cos)
            if date_obj:
                replacement_dates_objs.append(date_obj)
                replacement_dates_list.append(date_obj.strftime('%Y-%m-%d'))
        
        if replacement_dates_objs:
            replacement_date_range = {
                'min_date': min(replacement_dates_objs),
                'max_date': max(replacement_dates_objs),
                'span_days': (max(replacement_dates_objs) - min(replacement_dates_objs)).days,
                'unique_dates': len(set(d.date() for d in replacement_dates_objs))
            }
    
    return {
        'dataset_name': dataset_name,
        'total_rows': len(df),
        'placeholder_ndvi': placeholder_ndvi_count,
        'placeholder_summer': placeholder_summer_count,
        'total_placeholders': placeholder_ndvi_count + placeholder_summer_count,
        'needs_replacement_count': len(needs_replacement),
        'date_range': date_range_info,
        'replacement_date_range': replacement_date_range,
        'replacement_dates': replacement_dates_list
    }

def main():
    """Main analysis function."""
    feature_files = [
        'data/features/national_refuge_features.csv',
        'data/features/northern_bighorn_features.csv',
        'data/features/southern_bighorn_features.csv',
        'data/features/southern_gye_features.csv'
    ]
    
    print("=" * 70)
    print("AppEEARS Batch Estimation Analysis")
    print("=" * 70)
    print(f"\nBatching Parameters:")
    print(f"  Date buffer: ±{DATE_BUFFER_DAYS} days (total range width: {calculate_date_range_width(DATE_BUFFER_DAYS)} days)")
    print(f"  Max points per batch: {MAX_POINTS_PER_BATCH}")
    print(f"  Date range overlap logic: Points with overlapping ranges are grouped")
    print()
    
    results = []
    total_points = 0
    total_placeholders = 0
    all_replacement_dates = []
    
    for file_path in feature_files:
        result = analyze_feature_file(file_path)
        if result:
            results.append(result)
            total_points += result['total_rows']
            total_placeholders += result['total_placeholders']
            all_replacement_dates.extend(result['replacement_dates'])
            
            print(f"Dataset: {result['dataset_name']}")
            print(f"  Total rows: {result['total_rows']:,}")
            print(f"  Placeholder NDVI: {result['placeholder_ndvi']:,}")
            print(f"  Placeholder summer_integrated_ndvi: {result['placeholder_summer']:,}")
            print(f"  Total needing replacement: {result['needs_replacement_count']:,}")
            
            if result['date_range']:
                dr = result['date_range']
                print(f"  Date range: {dr['min_date'].date()} to {dr['max_date'].date()} ({dr['span_days']} days, {dr['unique_dates']} unique dates)")
            
            if result['replacement_date_range']:
                rdr = result['replacement_date_range']
                print(f"  Replacement date range: {rdr['min_date'].date()} to {rdr['max_date'].date()} ({rdr['span_days']} days, {rdr['unique_dates']} unique dates)")
            
            # Estimate batches for this dataset
            if result['replacement_dates']:
                estimated_batches = estimate_batches_for_points(
                    result['replacement_dates'],
                    date_buffer_days=DATE_BUFFER_DAYS,
                    max_points_per_batch=MAX_POINTS_PER_BATCH
                )
                print(f"  Estimated batches needed: {estimated_batches:,}")
                print(f"  Average points per batch: {result['needs_replacement_count'] / estimated_batches:.1f}" if estimated_batches > 0 else "  Average points per batch: N/A")
            print()
    
    # Overall summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total points across all datasets: {total_points:,}")
    print(f"Total placeholder NDVI values: {total_placeholders:,}")
    print(f"Percentage needing replacement: {total_placeholders/total_points*100:.1f}%" if total_points > 0 else "")
    print()
    
    # Estimate total batches
    if all_replacement_dates:
        # Remove duplicates and estimate
        unique_dates = list(set(all_replacement_dates))
        total_estimated_batches = estimate_batches_for_points(
            unique_dates,
            date_buffer_days=DATE_BUFFER_DAYS,
            max_points_per_batch=MAX_POINTS_PER_BATCH
        )
        
        # But we need to account for multiple points per date
        # Group by date to see distribution
        date_counts = defaultdict(int)
        for date_str in all_replacement_dates:
            date_counts[date_str] += 1
        
        print(f"Date Distribution Analysis:")
        print(f"  Unique dates needing replacement: {len(unique_dates):,}")
        print(f"  Total points needing replacement: {len(all_replacement_dates):,}")
        print(f"  Average points per date: {len(all_replacement_dates) / len(unique_dates):.1f}")
        print()
        
        # More accurate estimate: group by date first, then estimate batches
        # Since multiple points can share the same date, we need to account for that
        # The grouping algorithm will group points with overlapping date ranges
        # If many points share the same date, they'll all be in one batch (up to max_points_per_batch)
        
        # Estimate: for each unique date, calculate how many batches it would need
        # Points on the same date will definitely be in the same batch
        batches_by_date_group = {}
        for date_str, count in date_counts.items():
            # Points on same date will be grouped together
            batches_for_this_date = (count + MAX_POINTS_PER_BATCH - 1) // MAX_POINTS_PER_BATCH  # Ceiling division
            batches_by_date_group[date_str] = batches_for_this_date
        
        # Now estimate how date groups combine (points with overlapping ranges)
        # This is a simplified estimate - actual grouping is more complex
        estimated_total_batches = estimate_batches_for_points(
            list(date_counts.keys()),
            date_buffer_days=DATE_BUFFER_DAYS,
            max_points_per_batch=MAX_POINTS_PER_BATCH
        )
        
        # But we need to multiply by average batches per date group
        avg_batches_per_date = sum(batches_by_date_group.values()) / len(batches_by_date_group) if batches_by_date_group else 1
        
        # More accurate: simulate the actual grouping
        # Group dates by overlapping ranges, then for each group, calculate batches needed
        date_list = list(date_counts.keys())
        date_ranges = []
        for date_str in date_list:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_date = target_date - timedelta(days=DATE_BUFFER_DAYS)
            end_date = target_date + timedelta(days=DATE_BUFFER_DAYS)
            date_ranges.append((start_date, end_date, date_str, date_counts[date_str]))
        
        # Group dates with overlapping ranges
        date_groups = []
        used = set()
        
        for i, (start1, end1, date1, count1) in enumerate(date_ranges):
            if i in used:
                continue
            
            group = [(date1, count1)]
            used.add(i)
            
            for j, (start2, end2, date2, count2) in enumerate(date_ranges[i+1:], start=i+1):
                if j in used:
                    continue
                
                if start1 <= end2 and start2 <= end1:
                    group.append((date2, count2))
                    used.add(j)
            
            date_groups.append(group)
        
        # Calculate batches for each date group
        total_batches_estimate = 0
        for group in date_groups:
            total_points_in_group = sum(count for _, count in group)
            batches_for_group = (total_points_in_group + MAX_POINTS_PER_BATCH - 1) // MAX_POINTS_PER_BATCH
            total_batches_estimate += batches_for_group
        
        print(f"Batch Estimation:")
        print(f"  Date groups (overlapping ranges): {len(date_groups):,}")
        print(f"  Estimated total batches needed: {total_batches_estimate:,}")
        print(f"  Average points per batch: {len(all_replacement_dates) / total_batches_estimate:.1f}" if total_batches_estimate > 0 else "")
        print()
        print(f"  Without batching (old approach): {len(all_replacement_dates):,} requests")
        print(f"  With batching (new approach): {total_batches_estimate:,} requests")
        print(f"  Reduction: {(1 - total_batches_estimate / len(all_replacement_dates)) * 100:.1f}%" if len(all_replacement_dates) > 0 else "")
        print(f"  Speedup: {len(all_replacement_dates) / total_batches_estimate:.1f}x" if total_batches_estimate > 0 else "")

if __name__ == '__main__':
    main()
