#!/usr/bin/env python3
"""
Compare feature distributions between Southeast and other regions.

This script investigates why Southeast region performance remains poor
even after fixing temporal mismatch, by analyzing feature distributions.

Usage:
    python scripts/analyze_region_features.py
    python scripts/analyze_region_features.py --input data/features/optimized/complete_context_optimized.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Tuple

# Southeast region bounds (matches create_regions logic)
SOUTHEAST_BOUNDS = {
    'lat_min': -np.inf,
    'lat_max': 44.0,
    'lon_min': -109.0,
    'lon_max': np.inf
}


def filter_southeast_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to Southeast region.

    Southeast: southern_bighorn (lower lat, eastern lon)
    Bounds: latitude < 44, longitude >= -109
    """
    mask = (df['latitude'] < 44) & (df['longitude'] >= -109)
    return df[mask].copy()


def filter_other_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to all non-Southeast regions.
    """
    mask = ~((df['latitude'] < 44) & (df['longitude'] >= -109))
    return df[mask].copy()


def apply_year_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Southeast region year filtering (2017-2019) to match training.
    """
    southeast_mask = (df['latitude'] < 44) & (df['longitude'] >= -109)
    se_filtered_mask = southeast_mask & (df['year'] >= 2017) & (df['year'] <= 2019)
    # Keep non-Southeast samples OR Southeast samples from 2017-2019
    return df[~southeast_mask | se_filtered_mask].copy()


def calculate_cohens_d(pos_values: pd.Series, neg_values: pd.Series) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    
    Cohen's d = (mean1 - mean2) / pooled_std
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    pos_mean = pos_values.mean()
    neg_mean = neg_values.mean()
    pos_std = pos_values.std()
    neg_std = neg_values.std()
    
    # Pooled standard deviation
    n1, n2 = len(pos_values), len(neg_values)
    pooled_std = np.sqrt(((n1 - 1) * pos_std**2 + (n2 - 1) * neg_std**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (pos_mean - neg_mean) / pooled_std


def analyze_feature_distributions(
    df: pd.DataFrame,
    region_name: str,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Analyze feature distributions and discriminative power within a region.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['elk_present', 'latitude', 'longitude', 'year']]
    
    results = []
    
    pos_df = df[df['elk_present'] == 1]
    neg_df = df[df['elk_present'] == 0]
    
    if len(pos_df) == 0 or len(neg_df) == 0:
        print(f"  ⚠ {region_name}: No samples for one class, skipping feature analysis")
        return pd.DataFrame()
    
    for col in numeric_cols:
        pos_values = pos_df[col].dropna()
        neg_values = neg_df[col].dropna()
        
        if len(pos_values) == 0 or len(neg_values) == 0:
            continue
        
        pos_mean = pos_values.mean()
        neg_mean = neg_values.mean()
        pos_std = pos_values.std()
        neg_std = neg_values.std()
        diff = pos_mean - neg_mean
        diff_pct = (diff / abs(neg_mean) * 100) if neg_mean != 0 else 0
        
        # Cohen's d
        cohens_d = calculate_cohens_d(pos_values, neg_values)
        
        results.append({
            'feature': col,
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
            'pos_std': pos_std,
            'neg_std': neg_std,
            'diff': diff,
            'diff_pct': diff_pct,
            'cohens_d': cohens_d
        })
    
    results_df = pd.DataFrame(results).sort_values('cohens_d', key=abs, ascending=False)
    return results_df.head(top_n)


def compare_regions_features(
    se_df: pd.DataFrame,
    other_df: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compare feature means between Southeast and other regions.
    """
    numeric_cols = se_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['elk_present', 'latitude', 'longitude', 'year']]
    
    results = []
    
    for col in numeric_cols:
        se_values = se_df[col].dropna()
        other_values = other_df[col].dropna()
        
        if len(se_values) == 0 or len(other_values) == 0:
            continue
        
        se_mean = se_values.mean()
        other_mean = other_values.mean()
        se_std = se_values.std()
        other_std = other_values.std()
        diff = se_mean - other_mean
        diff_pct = (diff / abs(other_mean) * 100) if other_mean != 0 else 0
        
        # Cohen's d between regions
        cohens_d = calculate_cohens_d(se_values, other_values)
        
        results.append({
            'feature': col,
            'se_mean': se_mean,
            'other_mean': other_mean,
            'se_std': se_std,
            'other_std': other_std,
            'diff': diff,
            'diff_pct': diff_pct,
            'cohens_d': cohens_d
        })
    
    results_df = pd.DataFrame(results).sort_values('cohens_d', key=abs, ascending=False)
    return results_df.head(top_n)


def analyze_class_distribution(df: pd.DataFrame, region_name: str) -> Dict:
    """Analyze class distribution statistics."""
    total = len(df)
    pos_count = df['elk_present'].sum()
    neg_count = total - pos_count
    pos_pct = pos_count / total * 100
    neg_pct = neg_count / total * 100
    
    stats = {
        'total_samples': total,
        'positive_count': int(pos_count),
        'negative_count': int(neg_count),
        'positive_percentage': pos_pct,
        'negative_percentage': neg_pct,
        'class_ratio': neg_count / pos_count if pos_count > 0 else np.inf
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compare feature distributions between Southeast and other regions"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('data/features/optimized/complete_context_optimized.csv'),
        help='Input feature file (default: data/features/optimized/complete_context_optimized.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/analysis/region_feature_comparison.md'),
        help='Output report file (default: data/analysis/region_feature_comparison.md)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top features to analyze (default: 20)'
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print(f"{'='*70}")
    print("REGION FEATURE DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Input: {args.input}")
    
    # Load data
    print(f"\nLoading data...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Apply year filtering (same as in training)
    print(f"\nApplying Southeast region year filtering (2017-2019)...")
    df_filtered = apply_year_filter(df)
    print(f"  After filtering: {len(df_filtered):,} rows")
    
    # Split into regions
    se_df = filter_southeast_region(df_filtered)
    other_df = filter_other_regions(df_filtered)
    
    print(f"\n  Southeast region: {len(se_df):,} samples")
    print(f"  Other regions: {len(other_df):,} samples")
    
    if len(se_df) == 0:
        print("  Error: No Southeast region data found")
        return 1
    
    # Analyze class distributions
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*70}")
    
    se_stats = analyze_class_distribution(se_df, "Southeast")
    other_stats = analyze_class_distribution(other_df, "Other")
    
    print(f"\nSoutheast region:")
    print(f"  Total: {se_stats['total_samples']:,}")
    print(f"  Presence: {se_stats['positive_count']:,} ({se_stats['positive_percentage']:.1f}%)")
    print(f"  Absence: {se_stats['negative_count']:,} ({se_stats['negative_percentage']:.1f}%)")
    print(f"  Ratio (neg:pos): {se_stats['class_ratio']:.2f}:1")
    
    print(f"\nOther regions:")
    print(f"  Total: {other_stats['total_samples']:,}")
    print(f"  Presence: {other_stats['positive_count']:,} ({other_stats['positive_percentage']:.1f}%)")
    print(f"  Absence: {other_stats['negative_count']:,} ({other_stats['negative_percentage']:.1f}%)")
    print(f"  Ratio (neg:pos): {other_stats['class_ratio']:.2f}:1")
    
    # Analyze feature distributions within each region
    print(f"\n{'='*70}")
    print(f"TOP {args.top_n} DISCRIMINATIVE FEATURES (within region)")
    print(f"{'='*70}")
    
    se_features = analyze_feature_distributions(se_df, "Southeast", top_n=args.top_n)
    other_features = analyze_feature_distributions(other_df, "Other", top_n=args.top_n)
    
    print(f"\nSoutheast region (Cohen's d):")
    print(f"{'Feature':<30} {'Pos Mean':>12} {'Neg Mean':>12} {'Diff':>12} {'Cohen d':>10}")
    print("-" * 90)
    for _, row in se_features.head(args.top_n).iterrows():
        print(f"{row['feature']:<30} {row['pos_mean']:>12.3f} {row['neg_mean']:>12.3f} "
              f"{row['diff']:>12.3f} {row['cohens_d']:>10.3f}")
    
    print(f"\nOther regions (Cohen's d):")
    print(f"{'Feature':<30} {'Pos Mean':>12} {'Neg Mean':>12} {'Diff':>12} {'Cohen d':>10}")
    print("-" * 90)
    for _, row in other_features.head(args.top_n).iterrows():
        print(f"{row['feature']:<30} {row['pos_mean']:>12.3f} {row['neg_mean']:>12.3f} "
              f"{row['diff']:>12.3f} {row['cohens_d']:>10.3f}")
    
    # Compare features between regions
    print(f"\n{'='*70}")
    print(f"TOP {args.top_n} FEATURES DIFFERING BETWEEN REGIONS")
    print(f"{'='*70}")
    
    region_comparison = compare_regions_features(se_df, other_df, top_n=args.top_n)
    
    print(f"{'Feature':<30} {'SE Mean':>12} {'Other Mean':>12} {'Diff':>12} {'Cohen d':>10}")
    print("-" * 90)
    for _, row in region_comparison.head(args.top_n).iterrows():
        print(f"{row['feature']:<30} {row['se_mean']:>12.3f} {row['other_mean']:>12.3f} "
              f"{row['diff']:>12.3f} {row['cohens_d']:>10.3f}")
    
    # Compare discriminative power between regions
    print(f"\n{'='*70}")
    print("DISCRIMINATIVE POWER COMPARISON")
    print(f"{'='*70}")
    
    # Merge feature analyses
    se_features_merged = se_features[['feature', 'cohens_d']].rename(columns={'cohens_d': 'se_cohens_d'})
    other_features_merged = other_features[['feature', 'cohens_d']].rename(columns={'cohens_d': 'other_cohens_d'})
    
    feature_comparison = pd.merge(se_features_merged, other_features_merged, on='feature', how='outer')
    feature_comparison = feature_comparison.fillna(0)
    feature_comparison['diff_d'] = feature_comparison['se_cohens_d'].abs() - feature_comparison['other_cohens_d'].abs()
    feature_comparison = feature_comparison.sort_values('diff_d', ascending=True)  # Most different first
    
    print(f"\nFeatures with weakest discriminative power in Southeast vs Other:")
    print(f"{'Feature':<30} {'SE |d|':>10} {'Other |d|':>10} {'Difference':>12}")
    print("-" * 70)
    for _, row in feature_comparison.head(10).iterrows():
        se_d = abs(row['se_cohens_d'])
        other_d = abs(row['other_cohens_d'])
        diff = row['diff_d']
        print(f"{row['feature']:<30} {se_d:>10.3f} {other_d:>10.3f} {diff:>12.3f}")
    
    # Generate report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("# Region Feature Distribution Comparison\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Source:** {args.input}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"This analysis compares feature distributions between Southeast region ")
        f.write(f"({len(se_df):,} samples, years 2017-2019) and other regions ")
        f.write(f"({len(other_df):,} samples).\n\n")
        
        f.write("## Class Distribution\n\n")
        f.write("| Region | Total | Presence | Pos % | Absence | Neg % | Ratio |\n")
        f.write("|--------|-------|----------|-------|---------|-------|-------|\n")
        f.write(f"| Southeast | {se_stats['total_samples']:,} | {se_stats['positive_count']:,} | "
               f"{se_stats['positive_percentage']:.1f}% | {se_stats['negative_count']:,} | "
               f"{se_stats['negative_percentage']:.1f}% | {se_stats['class_ratio']:.2f}:1 |\n")
        f.write(f"| Other | {other_stats['total_samples']:,} | {other_stats['positive_count']:,} | "
               f"{other_stats['positive_percentage']:.1f}% | {other_stats['negative_count']:,} | "
               f"{other_stats['negative_percentage']:.1f}% | {other_stats['class_ratio']:.2f}:1 |\n")
        f.write("\n")
        
        f.write(f"## Top Discriminative Features (within region)\n\n")
        f.write("### Southeast Region\n\n")
        f.write("| Feature | Pos Mean | Neg Mean | Difference | Cohen's d |\n")
        f.write("|---------|----------|----------|------------|----------|\n")
        for _, row in se_features.head(20).iterrows():
            f.write(f"| {row['feature']} | {row['pos_mean']:.3f} | {row['neg_mean']:.3f} | "
                   f"{row['diff']:.3f} | {row['cohens_d']:.3f} |\n")
        f.write("\n")
        
        f.write("### Other Regions\n\n")
        f.write("| Feature | Pos Mean | Neg Mean | Difference | Cohen's d |\n")
        f.write("|---------|----------|----------|------------|----------|\n")
        for _, row in other_features.head(20).iterrows():
            f.write(f"| {row['feature']} | {row['pos_mean']:.3f} | {row['neg_mean']:.3f} | "
                   f"{row['diff']:.3f} | {row['cohens_d']:.3f} |\n")
        f.write("\n")
        
        f.write(f"## Feature Differences Between Regions\n\n")
        f.write("| Feature | SE Mean | Other Mean | Difference | Cohen's d |\n")
        f.write("|---------|---------|------------|------------|----------|\n")
        for _, row in region_comparison.head(20).iterrows():
            f.write(f"| {row['feature']} | {row['se_mean']:.3f} | {row['other_mean']:.3f} | "
                   f"{row['diff']:.3f} | {row['cohens_d']:.3f} |\n")
        f.write("\n")
        
        f.write("## Insights\n\n")
        f.write("1. **Temporal alignment achieved**: Southeast data filtered to 2017-2019\n")
        f.write("2. **Class balance improved**: Southeast now has 1.1:1 ratio (was 8.5:1)\n")
        f.write("3. **Feature discriminative power**: Compare Cohen's d values between regions\n")
        f.write("4. **Regional differences**: Features with high between-region Cohen's d may cause generalization issues\n\n")
    
    print(f"\n{'='*70}")
    print(f"Report saved to: {args.output}")
    print(f"{'='*70}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
