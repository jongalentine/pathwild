#!/usr/bin/env python3
"""
Analyze optimization options for AppEEARS batching.

Tests different combinations of:
- date_buffer_days (wider date ranges = more grouping opportunities)
- max_points_per_batch (more points per request = fewer requests)

Shows trade-offs and recommendations.
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
    """Reconstruct date from year, month, and day_of_year_sin/cos."""
    if pd.isna(year) or pd.isna(month):
        return None
    
    try:
        year_int = int(year)
        month_int = int(month)
        
        if not (1 <= month_int <= 12):
            return None
        
        if not (pd.isna(day_of_year_sin) or pd.isna(day_of_year_cos)):
            angle = math.atan2(day_of_year_sin, day_of_year_cos)
            if angle < 0:
                angle += 2 * math.pi
            day_of_year = (angle / (2 * math.pi)) * 365.25
            day_of_year = int(round(day_of_year))
            base_date = datetime(year_int, 1, 1)
            target_date = base_date + timedelta(days=day_of_year - 1)
            return target_date
        else:
            return datetime(year_int, month_int, 15)
    except (ValueError, TypeError, OverflowError):
        return None

def estimate_batches_for_points(points_with_dates, date_buffer_days=5, max_points_per_batch=100):
    """Estimate number of batches needed for a set of points."""
    if not points_with_dates:
        return 0, {}
    
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
        
        for j, (start2, end2, date2) in enumerate(point_ranges[i+1:], start=i+1):
            if j in used:
                continue
            
            if start1 <= end2 and start2 <= end1:
                if len(group) < max_points_per_batch:
                    group.append(date2)
                    used.add(j)
        
        groups.append(group)
    
    # Calculate statistics
    group_sizes = [len(g) for g in groups]
    stats = {
        'num_batches': len(groups),
        'avg_points_per_batch': sum(group_sizes) / len(groups) if groups else 0,
        'min_points_per_batch': min(group_sizes) if group_sizes else 0,
        'max_points_per_batch': max(group_sizes) if group_sizes else 0,
        'batches_at_capacity': sum(1 for s in group_sizes if s >= max_points_per_batch),
        'utilization_pct': (sum(group_sizes) / (len(groups) * max_points_per_batch) * 100) if groups else 0
    }
    
    return len(groups), stats

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
    
    # Get points that need NDVI replacement
    needs_replacement_mask = pd.Series([False] * len(df))
    
    if 'ndvi' in df.columns:
        needs_replacement_mask |= df['ndvi'].isin(NDVI_PLACEHOLDERS)
    
    if 'summer_integrated_ndvi' in df.columns:
        needs_replacement_mask |= (df['summer_integrated_ndvi'] == SUMMER_NDVI_PLACEHOLDER)
    
    # Extract dates for points needing replacement
    replacement_dates_list = []
    if needs_replacement_mask.any() and all(col in df.columns for col in ['year', 'month']):
        replacement_df = df[needs_replacement_mask].copy()
        
        for _, row in replacement_df.iterrows():
            year = row.get('year')
            month = row.get('month')
            day_of_year_sin = row.get('day_of_year_sin', np.nan)
            day_of_year_cos = row.get('day_of_year_cos', np.nan)
            
            date_obj = reconstruct_date_from_sin_cos(year, month, day_of_year_sin, day_of_year_cos)
            if date_obj:
                replacement_dates_list.append(date_obj.strftime('%Y-%m-%d'))
    
    return {
        'dataset_name': dataset_name,
        'total_rows': len(df),
        'placeholder_ndvi': placeholder_ndvi_count,
        'placeholder_summer': placeholder_summer_count,
        'total_placeholders': placeholder_ndvi_count + placeholder_summer_count,
        'needs_replacement_count': needs_replacement_mask.sum(),
        'replacement_dates': replacement_dates_list
    }

def main():
    """Main optimization analysis."""
    feature_files = [
        'data/features/national_refuge_features.csv',
        'data/features/northern_bighorn_features.csv',
        'data/features/southern_bighorn_features.csv',
        'data/features/southern_gye_features.csv'
    ]
    
    print("=" * 80)
    print("AppEEARS Batching Optimization Analysis")
    print("=" * 80)
    print()
    
    # Collect all replacement dates
    all_replacement_dates = []
    total_points_needing_replacement = 0
    
    for file_path in feature_files:
        result = analyze_feature_file(file_path)
        if result and result['replacement_dates']:
            all_replacement_dates.extend(result['replacement_dates'])
            total_points_needing_replacement += result['needs_replacement_count']
    
    if not all_replacement_dates:
        print("No placeholder NDVI values found. Nothing to optimize.")
        return
    
    print(f"Total points needing replacement: {total_points_needing_replacement:,}")
    print(f"Unique dates: {len(set(all_replacement_dates)):,}")
    print()
    
    # Test different parameter combinations
    date_buffer_options = [5, 7, 10, 14, 21, 30]
    batch_size_options = [100, 200, 500, 1000]
    
    print("Testing parameter combinations...")
    print()
    
    results = []
    
    for date_buffer in date_buffer_options:
        for batch_size in batch_size_options:
            # Group dates by overlapping ranges
            date_counts = defaultdict(int)
            for date_str in all_replacement_dates:
                date_counts[date_str] += 1
            
            # Estimate batches
            date_list = list(date_counts.keys())
            date_ranges = []
            for date_str in date_list:
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
                start_date = target_date - timedelta(days=date_buffer)
                end_date = target_date + timedelta(days=date_buffer)
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
            total_batches = 0
            total_points_in_batches = 0
            batches_at_capacity = 0
            
            for group in date_groups:
                total_points_in_group = sum(count for _, count in group)
                batches_for_group = (total_points_in_group + batch_size - 1) // batch_size  # Ceiling division
                total_batches += batches_for_group
                total_points_in_batches += total_points_in_group
                
                if total_points_in_group >= batch_size:
                    batches_at_capacity += 1
            
            avg_points_per_batch = total_points_in_batches / total_batches if total_batches > 0 else 0
            utilization = (avg_points_per_batch / batch_size * 100) if batch_size > 0 else 0
            
            results.append({
                'date_buffer': date_buffer,
                'batch_size': batch_size,
                'total_batches': total_batches,
                'avg_points_per_batch': avg_points_per_batch,
                'utilization_pct': utilization,
                'date_groups': len(date_groups),
                'batches_at_capacity': batches_at_capacity,
                'speedup': total_points_needing_replacement / total_batches if total_batches > 0 else 0
            })
    
    # Display results
    print(f"{'Date Buffer':<12} {'Batch Size':<12} {'Total Batches':<15} {'Avg/Batch':<12} {'Utilization':<12} {'Speedup':<10}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: (x['total_batches'], -x['utilization_pct'])):
        print(f"{r['date_buffer']:>3} days     {r['batch_size']:>4}        {r['total_batches']:>8,}        "
              f"{r['avg_points_per_batch']:>6.1f}      {r['utilization_pct']:>6.1f}%      {r['speedup']:>6.1f}x")
    
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    # Find best options
    best_by_batches = min(results, key=lambda x: x['total_batches'])
    best_by_utilization = max(results, key=lambda x: x['utilization_pct'])
    best_balanced = min([r for r in results if r['utilization_pct'] >= 70], 
                       key=lambda x: x['total_batches'], default=None)
    
    print("1. MINIMUM BATCHES (fewest API requests):")
    print(f"   date_buffer_days={best_by_batches['date_buffer']}, max_points_per_batch={best_by_batches['batch_size']}")
    print(f"   → {best_by_batches['total_batches']:,} batches ({best_by_batches['speedup']:.1f}x speedup)")
    print(f"   → {best_by_batches['utilization_pct']:.1f}% batch utilization")
    print()
    
    if best_balanced:
        print("2. BALANCED (good utilization + fewer batches):")
        print(f"   date_buffer_days={best_balanced['date_buffer']}, max_points_per_batch={best_balanced['batch_size']}")
        print(f"   → {best_balanced['total_batches']:,} batches ({best_balanced['speedup']:.1f}x speedup)")
        print(f"   → {best_balanced['utilization_pct']:.1f}% batch utilization")
        print()
    
    print("3. BEST UTILIZATION (most efficient use of batch capacity):")
    print(f"   date_buffer_days={best_by_utilization['date_buffer']}, max_points_per_batch={best_by_utilization['batch_size']}")
    print(f"   → {best_by_utilization['total_batches']:,} batches ({best_by_utilization['speedup']:.1f}x speedup)")
    print(f"   → {best_by_utilization['utilization_pct']:.1f}% batch utilization")
    print()
    
    print("=" * 80)
    print("TRADE-OFFS")
    print("=" * 80)
    print()
    print("Wider Date Buffer (±days):")
    print("  ✓ More points can be grouped together")
    print("  ✓ Fewer total batches needed")
    print("  ✗ Returns more data than needed (wider date range per request)")
    print("  ✗ Slightly more processing to find best match")
    print()
    print("Larger Batch Size (points per request):")
    print("  ✓ Fewer total batches needed")
    print("  ✓ Better utilization of API capacity")
    print("  ✗ Larger response payloads")
    print("  ✗ May hit API limits (check AppEEARS documentation)")
    print()
    print("Current defaults: date_buffer_days=5, max_points_per_batch=100")
    print()
    
    # Check AppEEARS API limits (if documented)
    print("NOTE: Check AppEEARS API documentation for:")
    print("  - Maximum coordinates per request")
    print("  - Maximum date range per request")
    print("  - Rate limiting constraints")
    print("  - Response size limits")

if __name__ == '__main__':
    main()
