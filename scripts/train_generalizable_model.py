#!/usr/bin/env python3
"""
Train a generalizable XGBoost model for elk presence prediction.

This script trains a model optimized for cross-region generalization by:
1. Excluding location-specific features (lat/lon)
2. Using spatial cross-validation (leave-one-region-out)
3. Applying stronger regularization to prevent overfitting
4. Focusing on habitat features that should generalize

Usage:
    python scripts/train_generalizable_model.py
    python scripts/train_generalizable_model.py --include-coords  # include lat/lon for comparison
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

import xgboost as xgb

# Features to exclude for better generalization
LOCATION_FEATURES = {'latitude', 'longitude'}

# Features that may cause region-specific overfitting
RISKY_FEATURES = {'year'}  # Year correlates with which dataset


def create_regions(df: pd.DataFrame) -> pd.Series:
    """
    Create region labels based on geographic coordinates.
    Roughly corresponds to the 4 source datasets.
    """
    regions = pd.Series('unknown', index=df.index)

    # Southwest: southern_gye area (lower lat, western lon)
    regions[(df['latitude'] < 43.5) & (df['longitude'] < -109)] = 'southwest'

    # Northwest: national_refuge area (higher lat, western lon)
    regions[(df['latitude'] >= 43.5) & (df['longitude'] < -109)] = 'northwest'

    # Southeast: southern_bighorn (lower lat, eastern lon)
    regions[(df['latitude'] < 44) & (df['longitude'] >= -109)] = 'southeast'

    # Northeast: northern_bighorn (higher lat, eastern lon)
    regions[(df['latitude'] >= 44) & (df['longitude'] >= -109)] = 'northeast'

    return regions


def spatial_cross_validation(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'elk_present',
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    reg_alpha: float = 1.0,
    reg_lambda: float = 5.0,
    min_child_weight: int = 10,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> dict:
    """
    Perform leave-one-region-out spatial cross-validation.
    """
    regions = create_regions(df)
    unique_regions = regions.unique()

    X = df[feature_cols]
    y = df[target_col]

    results = []
    all_predictions = []
    all_actuals = []

    print(f"\n{'='*70}")
    print("SPATIAL CROSS-VALIDATION (Leave-One-Region-Out)")
    print(f"{'='*70}")
    print(f"\nUsing {len(feature_cols)} features")
    print(f"Regions: {list(unique_regions)}")

    print(f"\n{'Test Region':<15} {'Train':>10} {'Test':>10} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 65)

    for region in unique_regions:
        train_mask = regions != region
        test_mask = regions == region

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            'region': region,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': acc,
            'f1': f1,
            'auc': auc,
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        })

        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)

        print(f"{region:<15} {len(X_train):>10,} {len(X_test):>10,} {acc:>8.1%} {f1:>8.1%} {auc:>8.1%}")

    # Overall metrics
    print("-" * 65)
    mean_acc = np.mean([r['accuracy'] for r in results])
    mean_f1 = np.mean([r['f1'] for r in results])
    mean_auc = np.mean([r['auc'] for r in results])

    # Weighted by test size
    total_test = sum(r['test_size'] for r in results)
    weighted_acc = sum(r['accuracy'] * r['test_size'] for r in results) / total_test
    weighted_f1 = sum(r['f1'] * r['test_size'] for r in results) / total_test

    print(f"{'MEAN':<15} {'':>10} {'':>10} {mean_acc:>8.1%} {mean_f1:>8.1%} {mean_auc:>8.1%}")
    print(f"{'WEIGHTED':<15} {'':>10} {'':>10} {weighted_acc:>8.1%} {weighted_f1:>8.1%}")

    return {
        'fold_results': results,
        'mean_accuracy': mean_acc,
        'mean_f1': mean_f1,
        'mean_auc': mean_auc,
        'weighted_accuracy': weighted_acc,
        'weighted_f1': weighted_f1,
        'all_predictions': all_predictions,
        'all_actuals': all_actuals
    }


def train_final_model(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'elk_present',
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    reg_alpha: float = 1.0,
    reg_lambda: float = 5.0,
    min_child_weight: int = 10,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> tuple:
    """
    Train final model on all data with regularization for generalization.
    """
    X = df[feature_cols]
    y = df[target_col]

    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*70}")

    print(f"\nHyperparameters (tuned for generalization):")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth} (shallow for generalization)")
    print(f"  learning_rate: {learning_rate}")
    print(f"  reg_alpha (L1): {reg_alpha}")
    print(f"  reg_lambda (L2): {reg_lambda}")
    print(f"  min_child_weight: {min_child_weight}")
    print(f"  subsample: {subsample}")
    print(f"  colsample_bytree: {colsample_bytree}")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss'
    )

    print(f"\nTraining on {len(X):,} samples...")
    model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 15 Feature Importance:")
    print(f"{'Feature':<35} {'Importance':>12}")
    print("-" * 48)
    for _, row in importance.head(15).iterrows():
        print(f"{row['feature']:<35} {row['importance']:>12.4f}")

    return model, importance


def main():
    parser = argparse.ArgumentParser(
        description="Train generalizable XGBoost model for elk presence prediction"
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('data/features/optimized/complete_context_optimized.csv'),
        help='Input feature file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('models'),
        help='Output directory'
    )
    parser.add_argument(
        '--include-coords',
        action='store_true',
        help='Include lat/lon coordinates (for comparison)'
    )
    parser.add_argument(
        '--include-year',
        action='store_true',
        help='Include year feature (may cause dataset-specific overfitting)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=4,
        help='Max tree depth (default: 4, shallow for generalization)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200,
        help='Number of trees (default: 200)'
    )

    args = parser.parse_args()

    print(f"{'='*70}")
    print("TRAINING GENERALIZABLE ELK PRESENCE MODEL")
    print(f"{'='*70}")
    print(f"Started at: {datetime.now()}")
    print(f"Input: {args.input}")

    # Load data
    print(f"\nLoading data...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Encode categorical columns
    label_encoders = {}
    for col in ['snow_crust_detected', 'land_cover_type']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna('unknown')
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Define feature columns
    exclude_cols = {'elk_present'}

    if not args.include_coords:
        exclude_cols.update(LOCATION_FEATURES)
        print(f"\n  Excluding location features: {LOCATION_FEATURES}")

    if not args.include_year:
        exclude_cols.update(RISKY_FEATURES)
        print(f"  Excluding risky features: {RISKY_FEATURES}")

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"  Using {len(feature_cols)} features")

    # Spatial cross-validation
    cv_results = spatial_cross_validation(
        df, feature_cols,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators
    )

    # Train final model
    model, importance = train_final_model(
        df, feature_cols,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators
    )

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "xgboost_generalizable"
    if args.include_coords:
        model_name += "_with_coords"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pickle with metadata
    pickle_file = args.output_dir / f"{model_name}_{timestamp}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'label_encoders': label_encoders,
            'cv_results': {
                'mean_accuracy': cv_results['mean_accuracy'],
                'mean_f1': cv_results['mean_f1'],
                'weighted_accuracy': cv_results['weighted_accuracy'],
                'fold_results': cv_results['fold_results']
            }
        }, f)
    print(f"\n  Model saved to: {pickle_file}")

    # Save as XGBoost native format
    json_file = args.output_dir / f"{model_name}_{timestamp}.json"
    model.save_model(str(json_file))
    print(f"  Model saved to: {json_file}")

    # Save feature importance
    importance_file = args.output_dir / f"{model_name}_{timestamp}_feature_importance.csv"
    importance.to_csv(importance_file, index=False)
    print(f"  Feature importance saved to: {importance_file}")

    # Save results summary
    results_file = args.output_dir / f"{model_name}_{timestamp}_results.json"
    results = {
        'timestamp': timestamp,
        'input_file': str(args.input),
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'features': feature_cols,
        'include_coords': args.include_coords,
        'include_year': args.include_year,
        'hyperparameters': {
            'max_depth': args.max_depth,
            'n_estimators': args.n_estimators,
            'learning_rate': 0.05,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'min_child_weight': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'spatial_cv': {
            'mean_accuracy': cv_results['mean_accuracy'],
            'mean_f1': cv_results['mean_f1'],
            'mean_auc': cv_results['mean_auc'],
            'weighted_accuracy': cv_results['weighted_accuracy'],
            'fold_results': cv_results['fold_results']
        },
        'feature_importance': importance.to_dict('records')
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {results_file}")

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nSpatial CV Performance (realistic for new regions):")
    print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.1%}")
    print(f"  Mean F1 Score: {cv_results['mean_f1']:.1%}")
    print(f"  Mean AUC-ROC:  {cv_results['mean_auc']:.1%}")

    target = 0.70
    if cv_results['mean_accuracy'] >= target:
        print(f"\n  ✓ TARGET MET: {cv_results['mean_accuracy']:.1%} >= {target:.0%}")
    else:
        gap = target - cv_results['mean_accuracy']
        print(f"\n  ✗ TARGET NOT MET: {cv_results['mean_accuracy']:.1%} < {target:.0%}")
        print(f"    Gap: {gap*100:.1f} percentage points")

    print(f"\n{'='*70}")
    print(f"Finished at: {datetime.now()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
