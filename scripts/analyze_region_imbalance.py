#!/usr/bin/env python3
"""
Analyze class imbalance in Southeast region.

This script investigates the severe class imbalance in the Southeast region
(84.6% accuracy, 9.1% F1, 7.3% recall) to understand root causes and inform
targeted fixes.

Usage:
    python scripts/analyze_region_imbalance.py
    python scripts/analyze_region_imbalance.py --input data/features/optimized/complete_context_optimized.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

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


def analyze_class_distribution(df: pd.DataFrame, region_name: str = "Southeast") -> dict:
    """Analyze class distribution statistics."""
    stats = {}
    
    total = len(df)
    pos_count = df['elk_present'].sum()
    neg_count = total - pos_count
    pos_pct = pos_count / total * 100
    neg_pct = neg_count / total * 100
    
    stats['total_samples'] = total
    stats['positive_count'] = int(pos_count)
    stats['negative_count'] = int(neg_count)
    stats['positive_percentage'] = pos_pct
    stats['negative_percentage'] = neg_pct
    stats['class_ratio'] = neg_count / pos_count if pos_count > 0 else np.inf
    
    print(f"\n{'='*70}")
    print(f"{region_name} Region - Class Distribution")
    print(f"{'='*70}")
    print(f"Total samples: {total:,}")
    print(f"Positive (elk_present=1): {pos_count:,} ({pos_pct:.1f}%)")
    print(f"Negative (elk_present=0): {neg_count:,} ({neg_pct:.1f}%)")
    print(f"Class ratio (neg:pos): {stats['class_ratio']:.2f}:1")
    
    return stats


def analyze_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze class distribution by month."""
    if 'month' not in df.columns:
        print("  ⚠ month column not found, skipping monthly analysis")
        return pd.DataFrame()
    
    monthly_stats = df.groupby(['month', 'elk_present']).size().unstack(fill_value=0)
    monthly_stats['total'] = monthly_stats.sum(axis=1)
    monthly_stats['pos_pct'] = (monthly_stats.get(1, 0) / monthly_stats['total'] * 100).round(1)
    monthly_stats['neg_pct'] = (monthly_stats.get(0, 0) / monthly_stats['total'] * 100).round(1)
    
    print(f"\n{'='*70}")
    print("Class Distribution by Month")
    print(f"{'='*70}")
    print(f"{'Month':<8} {'Total':>10} {'Positive':>12} {'Pos%':>8} {'Negative':>12} {'Neg%':>8}")
    print("-" * 70)
    
    for month in sorted(monthly_stats.index):
        total = int(monthly_stats.loc[month, 'total'])
        pos = int(monthly_stats.loc[month, 1]) if 1 in monthly_stats.columns else 0
        neg = int(monthly_stats.loc[month, 0]) if 0 in monthly_stats.columns else 0
        pos_pct = monthly_stats.loc[month, 'pos_pct']
        neg_pct = monthly_stats.loc[month, 'neg_pct']
        print(f"{month:<8} {total:>10,} {pos:>12,} {pos_pct:>7.1f}% {neg:>12,} {neg_pct:>7.1f}%")
    
    return monthly_stats


def analyze_feature_distributions(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Compare feature distributions between presence and absence classes."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'elk_present']
    
    results = []
    
    for col in numeric_cols:
        if col in ['latitude', 'longitude']:  # Skip location features for now
            continue
            
        pos_values = df[df['elk_present'] == 1][col].dropna()
        neg_values = df[df['elk_present'] == 0][col].dropna()
        
        if len(pos_values) == 0 or len(neg_values) == 0:
            continue
        
        pos_mean = pos_values.mean()
        neg_mean = neg_values.mean()
        pos_std = pos_values.std()
        neg_std = neg_values.std()
        diff = pos_mean - neg_mean
        diff_pct = (diff / abs(neg_mean) * 100) if neg_mean != 0 else 0
        
        # Cohen's d for effect size
        pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
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
    
    print(f"\n{'='*70}")
    print(f"Top {top_n} Features by Discriminative Power (Cohen's d)")
    print(f"{'='*70}")
    print(f"{'Feature':<30} {'Pos Mean':>12} {'Neg Mean':>12} {'Diff':>12} {'Cohen d':>10}")
    print("-" * 90)
    
    for _, row in results_df.head(top_n).iterrows():
        print(f"{row['feature']:<30} {row['pos_mean']:>12.3f} {row['neg_mean']:>12.3f} "
              f"{row['diff']:>12.3f} {row['cohens_d']:>10.3f}")
    
    return results_df


def check_data_quality(df: pd.DataFrame) -> dict:
    """Check for data quality issues."""
    issues = {}
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)
    high_missing = missing_pct[missing_pct > 5]
    
    if len(high_missing) > 0:
        issues['high_missing_values'] = high_missing.to_dict()
        print(f"\n{'='*70}")
        print("Data Quality Issues - High Missing Values (>5%)")
        print(f"{'='*70}")
        for col, pct in high_missing.items():
            print(f"  {col}: {missing[col]:,} ({pct:.1f}%)")
    
    # Outliers (using IQR method for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['elk_present', 'latitude', 'longitude']]
    
    outlier_counts = {}
    for col in numeric_cols[:10]:  # Check first 10 numeric columns to avoid too much output
        if df[col].notna().sum() == 0:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > len(df) * 0.01:  # More than 1% outliers
            outlier_counts[col] = outliers
    
    if outlier_counts:
        issues['outliers'] = outlier_counts
        print(f"\n{'='*70}")
        print("Data Quality Issues - High Outlier Count (>1% of data)")
        print(f"{'='*70}")
        for col, count in outlier_counts.items():
            print(f"  {col}: {count:,} outliers ({count/len(df)*100:.1f}%)")
    
    if not issues:
        print(f"\n{'='*70}")
        print("Data Quality Check")
        print(f"{'='*70}")
        print("  ✓ No major data quality issues detected")
    
    return issues


def generate_report(
    input_file: Path,
    output_file: Path,
    class_stats: dict,
    monthly_stats: pd.DataFrame,
    feature_stats: pd.DataFrame,
    data_quality: dict
):
    """Generate markdown report."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# Southeast Region Class Imbalance Analysis\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Source:** {input_file}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"The Southeast region shows severe class imbalance:\n")
        f.write(f"- **Total samples:** {class_stats['total_samples']:,}\n")
        f.write(f"- **Positive class:** {class_stats['positive_count']:,} ({class_stats['positive_percentage']:.1f}%)\n")
        f.write(f"- **Negative class:** {class_stats['negative_count']:,} ({class_stats['negative_percentage']:.1f}%)\n")
        f.write(f"- **Class ratio (neg:pos):** {class_stats['class_ratio']:.2f}:1\n\n")
        
        f.write("## Class Distribution by Month\n\n")
        if not monthly_stats.empty:
            f.write("| Month | Total | Positive | Pos % | Negative | Neg % |\n")
            f.write("|-------|-------|----------|-------|----------|-------|\n")
            for month in sorted(monthly_stats.index):
                total = int(monthly_stats.loc[month, 'total'])
                pos = int(monthly_stats.loc[month, 1]) if 1 in monthly_stats.columns else 0
                neg = int(monthly_stats.loc[month, 0]) if 0 in monthly_stats.columns else 0
                pos_pct = monthly_stats.loc[month, 'pos_pct']
                neg_pct = monthly_stats.loc[month, 'neg_pct']
                f.write(f"| {month} | {total:,} | {pos:,} | {pos_pct:.1f}% | {neg:,} | {neg_pct:.1f}% |\n")
        f.write("\n")
        
        f.write("## Top Discriminative Features (Cohen's d)\n\n")
        f.write("| Feature | Pos Mean | Neg Mean | Difference | Cohen's d |\n")
        f.write("|---------|----------|----------|------------|----------|\n")
        for _, row in feature_stats.head(15).iterrows():
            f.write(f"| {row['feature']} | {row['pos_mean']:.3f} | {row['neg_mean']:.3f} | "
                   f"{row['diff']:.3f} | {row['cohens_d']:.3f} |\n")
        f.write("\n")
        
        if data_quality:
            f.write("## Data Quality Issues\n\n")
            if 'high_missing_values' in data_quality:
                f.write("### High Missing Values (>5%)\n\n")
                for col, pct in data_quality['high_missing_values'].items():
                    f.write(f"- **{col}**: {pct:.1f}% missing\n")
                f.write("\n")
            
            if 'outliers' in data_quality:
                f.write("### High Outlier Count (>1%)\n\n")
                for col, count in data_quality['outliers'].items():
                    f.write(f"- **{col}**: {count:,} outliers\n")
                f.write("\n")
        else:
            f.write("## Data Quality\n\n")
            f.write("✓ No major data quality issues detected.\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **Class Weighting**: Use `--class-weight auto` to automatically balance classes\n")
        f.write("2. **Threshold Optimization**: Use `--optimize-threshold` to find per-region optimal thresholds\n")
        f.write("3. **Region-Specific Analysis**: Consider if Southeast region needs different treatment\n")
        f.write("4. **Data Collection**: Investigate if Southeast region has different data collection methodology\n\n")
    
    print(f"\n{'='*70}")
    print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze class imbalance in Southeast region"
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
        default=Path('data/analysis/southeast_region_analysis.md'),
        help='Output report file (default: data/analysis/southeast_region_analysis.md)'
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print(f"{'='*70}")
    print("SOUTHEAST REGION CLASS IMBALANCE ANALYSIS")
    print(f"{'='*70}")
    print(f"Input: {args.input}")
    
    # Load data
    print(f"\nLoading data...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Filter to Southeast region
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("  Error: latitude and longitude columns required for region filtering")
        return 1
    
    se_df = filter_southeast_region(df)
    print(f"  Southeast region: {len(se_df):,} samples ({len(se_df)/len(df)*100:.1f}% of total)")
    
    if len(se_df) == 0:
        print("  Error: No Southeast region data found")
        return 1
    
    # Perform analysis
    class_stats = analyze_class_distribution(se_df)
    monthly_stats = analyze_by_month(se_df)
    feature_stats = analyze_feature_distributions(se_df)
    data_quality = check_data_quality(se_df)
    
    # Generate report
    generate_report(
        args.input,
        args.output,
        class_stats,
        monthly_stats,
        feature_stats,
        data_quality
    )
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
