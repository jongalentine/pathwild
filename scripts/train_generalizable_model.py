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
from typing import Optional

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
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


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost binary classification.
    
    For binary classification, scale_pos_weight is the ratio of
    negative class samples to positive class samples.
    
    Args:
        y: Target series (binary: 0 or 1)
    
    Returns:
        scale_pos_weight value for XGBoost
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    
    if pos_count == 0:
        return 1.0  # Default if no positive samples
    
    return neg_count / pos_count


def find_optimal_threshold(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """
    Find optimal classification threshold that maximizes F1 score.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
    
    Returns:
        Optimal threshold value
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def temporal_train_test_split(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'elk_present',
    include_coords: bool = False,
    **model_kwargs
) -> dict:
    """
    Split data temporally: train on 2006-2016, test on 2017-2019, exclude 2020.

    Args:
        df: Input DataFrame (must have 'year' column)
        feature_cols: List of feature column names
        target_col: Target column name
        include_coords: Whether lat/lon are included in features
        **model_kwargs: Passed to XGBClassifier

    Returns:
        Dictionary with evaluation metrics and the trained evaluation model
    """
    print(f"\n{'='*70}")
    print("TEMPORAL TRAIN/TEST SPLIT")
    print(f"{'='*70}")

    # Exclude 2020
    df_filtered = df[df['year'] != 2020].copy()
    n_excluded = len(df) - len(df_filtered)
    print(f"\nExcluded {n_excluded:,} rows from year 2020")

    train_mask = df_filtered['year'] <= 2016
    test_mask = (df_filtered['year'] >= 2017) & (df_filtered['year'] <= 2019)

    X_train = df_filtered.loc[train_mask, feature_cols]
    y_train = df_filtered.loc[train_mask, target_col]
    X_test = df_filtered.loc[test_mask, feature_cols]
    y_test = df_filtered.loc[test_mask, target_col]

    # Print split statistics
    print(f"\n{'Split':<10} {'Rows':>10} {'Presence':>10} {'Absence':>10} {'Pres %':>8}")
    print("-" * 52)
    for name, y in [('Train', y_train), ('Test', y_test)]:
        pos = int(y.sum())
        neg = len(y) - pos
        pct = pos / len(y) * 100 if len(y) > 0 else 0
        print(f"{name:<10} {len(y):>10,} {pos:>10,} {neg:>10,} {pct:>7.1f}%")

    # Print year breakdown
    print(f"\nYear breakdown:")
    for year in sorted(df_filtered['year'].unique()):
        subset = df_filtered[df_filtered['year'] == year]
        pos = int(subset[target_col].sum())
        split = 'train' if year <= 2016 else 'test'
        print(f"  {year}: {len(subset):>8,} rows, {pos:>6,} presence ({split})")

    # Train evaluation model on training set only
    random_state = model_kwargs.pop('random_state', 42)
    scale_pos_weight = model_kwargs.pop('scale_pos_weight', None)
    if scale_pos_weight is None:
        scale_pos_weight = calculate_scale_pos_weight(y_train)

    model = xgb.XGBClassifier(
        n_estimators=model_kwargs.get('n_estimators', 200),
        max_depth=model_kwargs.get('max_depth', 4),
        learning_rate=model_kwargs.get('learning_rate', 0.05),
        reg_alpha=model_kwargs.get('reg_alpha', 1.0),
        reg_lambda=model_kwargs.get('reg_lambda', 5.0),
        min_child_weight=model_kwargs.get('min_child_weight', 10),
        subsample=model_kwargs.get('subsample', 0.8),
        colsample_bytree=model_kwargs.get('colsample_bytree', 0.8),
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss'
    )

    optimize_threshold = model_kwargs.pop('optimize_threshold', False)
    threshold_override = model_kwargs.pop('threshold_override', None)

    print(f"\nTraining evaluation model on {len(X_train):,} samples...")
    if threshold_override is not None:
        model.fit(X_train, y_train)
        threshold = threshold_override
        print(f"  Using manual threshold: {threshold:.3f}")
    elif optimize_threshold:
        # Hold out validation set from training data for threshold tuning
        from sklearn.model_selection import train_test_split
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )
        model.fit(X_train_fit, y_train_fit)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, y_val_proba)
        print(f"  Optimal threshold from validation set: {threshold:.3f}")
    else:
        model.fit(X_train, y_train)
        threshold = 0.5

    # Evaluate on test set
    y_proba = model.predict_proba(X_test)[:, 1]
    if threshold_override is not None or optimize_threshold:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*70}")
    print("TEMPORAL TEST SET RESULTS (2017-2019)")
    print(f"{'='*70}")
    print(f"\n  Accuracy:  {acc:.1%}")
    print(f"  F1 Score:  {f1:.1%}")
    print(f"  AUC-ROC:   {auc:.1%}")
    print(f"  Precision: {prec:.1%}")
    print(f"  Recall:    {rec:.1%}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted Absent  Predicted Present")
    print(f"  Actual Absent   {cm[0][0]:>10,}       {cm[0][1]:>10,}")
    print(f"  Actual Present  {cm[1][0]:>10,}       {cm[1][1]:>10,}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['absent', 'present']))

    # Feature importance from evaluation model
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"Top 15 Feature Importance (eval model):")
    print(f"{'Feature':<35} {'Importance':>12}")
    print("-" * 48)
    for _, row in importance.head(15).iterrows():
        print(f"{row['feature']:<35} {row['importance']:>12.4f}")

    target = 0.70
    if acc >= target:
        print(f"\n  TARGET MET: {acc:.1%} >= {target:.0%}")
    else:
        gap = target - acc
        print(f"\n  TARGET NOT MET: {acc:.1%} < {target:.0%}")
        print(f"    Gap: {gap*100:.1f} percentage points")

    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'precision': prec,
        'recall': rec,
        'confusion_matrix': cm.tolist(),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_importance': importance,
        'df_no_2020': df_filtered,
    }


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
    random_state: int = 42,
    scale_pos_weight: Optional[float] = None,
    optimize_threshold: bool = False,
    early_stopping_rounds: Optional[int] = None
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
    if scale_pos_weight is not None:
        print(f"Using scale_pos_weight: {scale_pos_weight:.3f}")
    print("-" * 65)

    optimal_thresholds = {}

    for region in unique_regions:
        train_mask = regions != region
        test_mask = regions == region

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Calculate or use provided scale_pos_weight
        fold_scale_pos_weight = scale_pos_weight
        if fold_scale_pos_weight is None:
            fold_scale_pos_weight = calculate_scale_pos_weight(y_train)

        # Log class distribution
        train_pos = y_train.sum()
        train_neg = len(y_train) - train_pos
        train_pos_pct = train_pos / len(y_train) * 100
        print(f"\n{region} - Train set: {train_pos:,} pos ({train_pos_pct:.1f}%), {train_neg:,} neg, scale_pos_weight={fold_scale_pos_weight:.3f}")

        # Prepare eval_set for early stopping and threshold optimization if requested
        eval_set = None
        X_train_fit, X_val, y_train_fit, y_val = None, None, None, None
        
        if early_stopping_rounds is not None or optimize_threshold:
            # Use 20% of training data as validation set
            from sklearn.model_selection import train_test_split
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
            )
            if early_stopping_rounds is not None:
                eval_set = [(X_val, y_val)]
        
        if X_train_fit is not None:
            fit_X, fit_y = X_train_fit, y_train_fit
        else:
            fit_X, fit_y = X_train, y_train

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=fold_scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss',
            early_stopping_rounds=early_stopping_rounds
        )

        if eval_set is not None:
            model.fit(fit_X, fit_y, eval_set=eval_set, verbose=False)
        else:
            model.fit(fit_X, fit_y)

        y_proba = model.predict_proba(X_test)[:, 1]

        # Optimize threshold if requested (use validation set if available, otherwise training set)
        if optimize_threshold:
            if X_val is not None and y_val is not None:
                # Use validation set for threshold optimization
                y_val_proba = model.predict_proba(X_val)[:, 1]
                threshold = find_optimal_threshold(y_val, y_val_proba)
            else:
                # Fall back to training set
                y_train_proba = model.predict_proba(fit_X)[:, 1]
                threshold = find_optimal_threshold(fit_y, y_train_proba)
            optimal_thresholds[region] = threshold
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
            optimal_thresholds[region] = 0.5  # Default threshold

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
            'recall': recall_score(y_test, y_pred),
            'scale_pos_weight': fold_scale_pos_weight,
            'threshold': optimal_thresholds[region],
            'train_pos_count': int(train_pos),
            'train_neg_count': int(train_neg)
        })

        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)

        threshold_str = f" (t={optimal_thresholds[region]:.3f})" if optimize_threshold else ""
        print(f"{region:<15} {len(X_train):>10,} {len(X_test):>10,} {acc:>8.1%} {f1:>8.1%} {auc:>8.1%}{threshold_str}")

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
        'all_actuals': all_actuals,
        'optimal_thresholds': optimal_thresholds
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
    random_state: int = 42,
    scale_pos_weight: Optional[float] = None,
    early_stopping_rounds: Optional[int] = None
) -> tuple:
    """
    Train final model on all data with regularization for generalization.
    """
    X = df[feature_cols]
    y = df[target_col]

    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*70}")

    # Calculate scale_pos_weight if not provided
    if scale_pos_weight is None:
        scale_pos_weight = calculate_scale_pos_weight(y)

    print(f"\nHyperparameters (tuned for generalization):")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth} (shallow for generalization)")
    print(f"  learning_rate: {learning_rate}")
    print(f"  reg_alpha (L1): {reg_alpha}")
    print(f"  reg_lambda (L2): {reg_lambda}")
    print(f"  min_child_weight: {min_child_weight}")
    print(f"  subsample: {subsample}")
    print(f"  colsample_bytree: {colsample_bytree}")
    print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

    # Log class distribution
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    pos_pct = pos_count / len(y) * 100
    print(f"\nClass distribution: {pos_count:,} positive ({pos_pct:.1f}%), {neg_count:,} negative")

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=early_stopping_rounds
    )

    print(f"\nTraining on {len(X):,} samples...")
    if early_stopping_rounds is not None:
        # Use 20% of data as validation set for early stopping
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
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


def hyperparameter_search(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'elk_present',
    scale_pos_weight: Optional[float] = None,
    optimize_threshold: bool = False
) -> dict:
    """
    Perform grid search on hyperparameters using spatial cross-validation.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        scale_pos_weight: Class weight parameter
        optimize_threshold: Whether to optimize threshold
    
    Returns:
        Dictionary with best hyperparameters and comparison results
    """
    print(f"\n{'='*70}")
    print("HYPERPARAMETER SEARCH (Grid Search)")
    print(f"{'='*70}")
    
    # Define hyperparameter grid
    param_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [5, 10, 20]
    }
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    print(f"\nSearching {total_combinations} hyperparameter combinations...")
    print(f"Using spatial cross-validation for each combination (may be slow)")
    print(f"\nHyperparameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    best_score = 0.0
    best_params = None
    all_results = []
    
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for i, combination in enumerate(itertools.product(*param_values), 1):
        params = dict(zip(param_names, combination))
        
        print(f"\n[{i}/{total_combinations}] Testing: {params}")
        
        # Run spatial CV with these parameters
        cv_results = spatial_cross_validation(
            df, feature_cols, target_col=target_col,
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            min_child_weight=params['min_child_weight'],
            scale_pos_weight=scale_pos_weight,
            optimize_threshold=optimize_threshold
        )
        
        # Use weighted F1 as the scoring metric (primary concern is F1 improvement)
        score = cv_results['weighted_f1']
        
        all_results.append({
            'params': params,
            'weighted_f1': score,
            'mean_f1': cv_results['mean_f1'],
            'mean_accuracy': cv_results['mean_accuracy'],
            'mean_auc': cv_results['mean_auc'],
            'weighted_accuracy': cv_results['weighted_accuracy']
        })
        
        print(f"  Result: weighted_f1={score:.1%}, mean_f1={cv_results['mean_f1']:.1%}, mean_acc={cv_results['mean_accuracy']:.1%}")
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"  ✓ New best! (weighted_f1={score:.1%})")
    
    print(f"\n{'='*70}")
    print("HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"\nBest hyperparameters (by weighted F1):")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nBest weighted F1: {best_score:.1%}")
    
    # Show top 5 results
    all_results_sorted = sorted(all_results, key=lambda x: x['weighted_f1'], reverse=True)
    print(f"\nTop 5 configurations:")
    print(f"{'Rank':<6} {'max_depth':<12} {'n_est':<8} {'lr':<8} {'min_child':<12} {'Weighted F1':<12} {'Mean F1':<10} {'Mean Acc':<10}")
    print("-" * 90)
    for rank, result in enumerate(all_results_sorted[:5], 1):
        p = result['params']
        print(f"{rank:<6} {p['max_depth']:<12} {p['n_estimators']:<8} {p['learning_rate']:<8} "
              f"{p['min_child_weight']:<12} {result['weighted_f1']:<12.1%} "
              f"{result['mean_f1']:<10.1%} {result['mean_accuracy']:<10.1%}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results_sorted
    }


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
    parser.add_argument(
        '--class-weight',
        type=str,
        default='none',
        choices=['none', 'auto', 'balanced'],
        help='Class weight strategy: "none" (default), "auto" (calculate from data), "balanced" (sklearn balanced)'
    )
    parser.add_argument(
        '--optimize-threshold',
        action='store_true',
        help='Optimize classification threshold per region to maximize F1 score'
    )
    parser.add_argument(
        '--early-stopping-rounds',
        type=int,
        default=None,
        help='Number of rounds for early stopping (uses 20%% validation split). Default: None (no early stopping)'
    )
    parser.add_argument(
        '--hyperparameter-search',
        action='store_true',
        help='Perform grid search on hyperparameters using spatial CV (may be slow)'
    )
    parser.add_argument(
        '--temporal-split',
        action='store_true',
        help='Use temporal train/test split (train: 2006-2016, test: 2017-2019, exclude 2020) instead of spatial CV'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Override classification threshold (0-1). Overrides --optimize-threshold.'
    )
    args = parser.parse_args()

    if args.temporal_split and args.hyperparameter_search:
        parser.error("--temporal-split and --hyperparameter-search are mutually exclusive")

    print(f"{'='*70}")
    print("TRAINING GENERALIZABLE ELK PRESENCE MODEL")
    print(f"{'='*70}")
    print(f"Started at: {datetime.now()}")
    print(f"Input: {args.input}")

    # Load data
    print(f"\nLoading data...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Filter Southeast region to years 2017-2019 to align presence and absence temporally
    # This addresses the 6-year temporal gap discovered in analysis
    # Skip when using temporal split - years are already separated, and keeping
    # pre-2017 SE absence data in training helps the model learn SE geography
    if args.temporal_split:
        print(f"\n  Skipping Southeast year filter (temporal split separates years already)")
    elif 'latitude' in df.columns and 'longitude' in df.columns and 'year' in df.columns:
        southeast_mask = (df['latitude'] < 44) & (df['longitude'] >= -109)
        before_filter = len(df[southeast_mask])

        # Keep Southeast samples only from years 2017-2019
        se_filtered_mask = southeast_mask & (df['year'] >= 2017) & (df['year'] <= 2019)
        se_removed = before_filter - se_filtered_mask.sum()

        # Apply filter: keep non-Southeast samples OR Southeast samples from 2017-2019
        df = df[~southeast_mask | se_filtered_mask].copy()

        print(f"\n  Applied Southeast region year filtering (2017-2019):")
        print(f"    Southeast samples before: {before_filter:,}")
        print(f"    Southeast samples after: {se_filtered_mask.sum():,}")
        print(f"    Removed {se_removed:,} Southeast samples from years outside 2017-2019")
        print(f"    Total samples after filter: {len(df):,}")

    # Encode categorical columns
    label_encoders = {}
    for col in ['snow_crust_detected', 'land_cover_type', 'rut_phase']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna('unknown')
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Define feature columns
    target_col = 'elk_present'
    exclude_cols = {target_col}

    if not args.include_coords:
        exclude_cols.update(LOCATION_FEATURES)
        print(f"\n  Excluding location features: {LOCATION_FEATURES}")

    if not args.include_year:
        exclude_cols.update(RISKY_FEATURES)
        print(f"  Excluding risky features: {RISKY_FEATURES}")

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"  Using {len(feature_cols)} features")

    # Calculate scale_pos_weight based on class weight strategy
    scale_pos_weight = None
    if args.class_weight == 'auto':
        scale_pos_weight = calculate_scale_pos_weight(df[target_col])
        print(f"\n  Using auto-calculated scale_pos_weight: {scale_pos_weight:.3f}")
    elif args.class_weight == 'balanced':
        # sklearn balanced: n_samples / (n_classes * np.bincount(y))
        # For XGBoost binary: scale_pos_weight = neg_count / pos_count (same as auto)
        scale_pos_weight = calculate_scale_pos_weight(df[target_col])
        print(f"\n  Using balanced scale_pos_weight: {scale_pos_weight:.3f}")
    else:
        print(f"\n  Not using class weights (scale_pos_weight=1.0)")

    if args.optimize_threshold:
        print(f"  Threshold optimization: ENABLED (will optimize per region)")

    # Hyperparameter search if requested (before final CV)
    search_results = None
    learning_rate = 0.05  # Default
    min_child_weight = 10  # Default

    if args.hyperparameter_search:
        search_results = hyperparameter_search(
            df, feature_cols, target_col=target_col,
            scale_pos_weight=scale_pos_weight,
            optimize_threshold=args.optimize_threshold
        )

        # Update hyperparameters with best from search
        args.max_depth = search_results['best_params']['max_depth']
        args.n_estimators = search_results['best_params']['n_estimators']
        learning_rate = search_results['best_params']['learning_rate']
        min_child_weight = search_results['best_params']['min_child_weight']
        print(f"\n  Using best hyperparameters from search:")
        print(f"    max_depth: {args.max_depth}, n_estimators: {args.n_estimators}")
        print(f"    learning_rate: {learning_rate}, min_child_weight: {min_child_weight}")

    model_kwargs = dict(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
    )

    if args.temporal_split:
        # Temporal evaluation
        temporal_results = temporal_train_test_split(
            df, feature_cols, target_col=target_col,
            include_coords=args.include_coords,
            optimize_threshold=args.optimize_threshold,
            threshold_override=args.threshold,
            **model_kwargs
        )

        # Train production model on all data minus 2020
        df_production = temporal_results['df_no_2020']
        model, importance = train_final_model(
            df_production, feature_cols,
            **model_kwargs,
            early_stopping_rounds=args.early_stopping_rounds
        )

        cv_results = {
            'mean_accuracy': temporal_results['accuracy'],
            'mean_f1': temporal_results['f1'],
            'mean_auc': temporal_results['auc'],
            'weighted_accuracy': temporal_results['accuracy'],
            'weighted_f1': temporal_results['f1'],
            'fold_results': [{
                'region': 'temporal_2017_2019',
                'train_size': temporal_results['train_size'],
                'test_size': temporal_results['test_size'],
                'accuracy': temporal_results['accuracy'],
                'f1': temporal_results['f1'],
                'auc': temporal_results['auc'],
                'precision': temporal_results['precision'],
                'recall': temporal_results['recall'],
            }]
        }
    else:
        # Spatial cross-validation
        cv_results = spatial_cross_validation(
            df, feature_cols,
            optimize_threshold=args.optimize_threshold,
            early_stopping_rounds=args.early_stopping_rounds,
            **model_kwargs
        )

        # Train final model on all data
        model, importance = train_final_model(
            df, feature_cols,
            early_stopping_rounds=args.early_stopping_rounds,
            **model_kwargs
        )

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "xgboost_generalizable"
    if args.include_coords:
        model_name += "_with_coords"
    if args.temporal_split:
        model_name += "_temporal"

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
    try:
        model.save_model(str(json_file))
        print(f"  Model saved to: {json_file}")
    except (TypeError, AttributeError) as e:
        # XGBoost 2.x may have issues with save_model on sklearn wrapper
        # Try using the underlying booster instead
        try:
            model.get_booster().save_model(str(json_file))
            print(f"  Model saved to: {json_file} (using booster)")
        except Exception as e2:
            print(f"  Warning: Could not save model in JSON format: {e2}")
            print(f"    Model is still available in pickle format: {pickle_file}")

    # Save feature importance
    importance_file = args.output_dir / f"{model_name}_{timestamp}_feature_importance.csv"
    importance.to_csv(importance_file, index=False)
    print(f"  Feature importance saved to: {importance_file}")

    # Save results summary
    eval_method = 'temporal_split' if args.temporal_split else 'spatial_cv'
    results_file = args.output_dir / f"{model_name}_{timestamp}_results.json"
    results = {
        'timestamp': timestamp,
        'input_file': str(args.input),
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'features': feature_cols,
        'include_coords': args.include_coords,
        'include_year': args.include_year,
        'evaluation_method': eval_method,
        'hyperparameters': {
            'max_depth': args.max_depth,
            'n_estimators': args.n_estimators,
            'learning_rate': learning_rate,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'min_child_weight': min_child_weight,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight if scale_pos_weight else 1.0,
            'optimize_threshold': args.optimize_threshold,
            'early_stopping_rounds': args.early_stopping_rounds,
            'hyperparameter_search': args.hyperparameter_search
        },
        eval_method: {
            'mean_accuracy': cv_results['mean_accuracy'],
            'mean_f1': cv_results['mean_f1'],
            'mean_auc': cv_results['mean_auc'],
            'weighted_accuracy': cv_results['weighted_accuracy'],
            'weighted_f1': cv_results['weighted_f1'],
            'fold_results': cv_results['fold_results']
        },
        'feature_importance': importance.to_dict('records')
    }

    if not args.temporal_split:
        results[eval_method]['optimal_thresholds'] = cv_results.get('optimal_thresholds', {})

    if args.temporal_split:
        results['temporal_split']['confusion_matrix'] = temporal_results['confusion_matrix']

    # Add hyperparameter search results if available
    if search_results:
        results['hyperparameter_search'] = {
            'best_params': search_results['best_params'],
            'best_score': search_results['best_score'],
            'top_5_results': search_results['all_results'][:5]
        }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {results_file}")

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if args.temporal_split:
        print(f"\nTemporal Split Performance (train: <=2016, test: 2017-2019):")
    else:
        print(f"\nSpatial CV Performance (realistic for new regions):")
    print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.1%}")
    print(f"  Mean F1 Score: {cv_results['mean_f1']:.1%}")
    print(f"  Mean AUC-ROC:  {cv_results['mean_auc']:.1%}")

    target = 0.70
    if cv_results['mean_accuracy'] >= target:
        print(f"\n  TARGET MET: {cv_results['mean_accuracy']:.1%} >= {target:.0%}")
    else:
        gap = target - cv_results['mean_accuracy']
        print(f"\n  TARGET NOT MET: {cv_results['mean_accuracy']:.1%} < {target:.0%}")
        print(f"    Gap: {gap*100:.1f} percentage points")

    print(f"\n{'='*70}")
    print(f"Finished at: {datetime.now()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
