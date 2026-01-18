#!/usr/bin/env python3
"""
Train XGBoost model for elk presence prediction.

This script trains an XGBoost classifier on the optimized feature set with:
- Temporal train/test split (train on earlier years, test on later years)
- Proper handling of categorical features
- Comprehensive evaluation metrics
- Model persistence and logging

Usage:
    python scripts/train_xgboost_model.py
    python scripts/train_xgboost_model.py --input data/features/optimized/complete_context_optimized.csv
    python scripts/train_xgboost_model.py --test-years 2015 2016
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import sys

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


def load_and_prepare_data(
    input_file: Path,
    test_years: list = None,
    test_size: float = 0.2,
    split_type: str = 'temporal',
    random_state: int = 42
) -> tuple:
    """
    Load data and prepare train/test splits.

    Args:
        input_file: Path to feature CSV
        test_years: Years to use for test set (temporal split only)
        test_size: Fraction for test set (stratified split only)
        split_type: 'temporal' or 'stratified'
        random_state: Random seed for reproducibility
    """
    from sklearn.model_selection import train_test_split

    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Identify feature columns
    target_col = 'elk_present'
    exclude_cols = {target_col}

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    print(f"  Categorical columns: {categorical_cols}")

    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns and col not in exclude_cols:
            le = LabelEncoder()
            # Handle NaN values
            df[col] = df[col].fillna('unknown')
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"    Encoded {col}: {len(le.classes_)} classes")

    # Feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"  Feature columns: {len(feature_cols)}")

    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle missing values (XGBoost can handle NaN, but let's be explicit)
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"  Missing values in features:")
        for col in missing_counts[missing_counts > 0].index:
            print(f"    {col}: {missing_counts[col]:,} ({missing_counts[col]/len(X)*100:.2f}%)")

    if split_type == 'stratified':
        # Stratified random split
        print(f"\n  Stratified random split:")
        print(f"    Test size: {test_size*100:.0f}%")
        print(f"    Random state: {random_state}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        print(f"    Train set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"    Test set:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    else:
        # Temporal split
        if test_years is None:
            # Default: use last 2 years for testing
            all_years = sorted(df['year'].unique())
            test_years = all_years[-2:] if len(all_years) > 2 else all_years[-1:]

        print(f"\n  Temporal split:")
        print(f"    Test years: {test_years}")

        train_mask = ~df['year'].isin(test_years)
        test_mask = df['year'].isin(test_years)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        print(f"    Train set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"    Test set:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"    Train years: {sorted(df[train_mask]['year'].unique())}")
        print(f"    Test years:  {sorted(df[test_mask]['year'].unique())}")

    # Class balance
    print(f"\n  Class balance:")
    print(f"    Train - Presence: {y_train.mean()*100:.1f}%")
    print(f"    Test  - Presence: {y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test, feature_cols, label_encoders


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 50,
    random_state: int = 42
) -> tuple:
    """
    Train XGBoost classifier with early stopping.
    """
    print(f"\n{'='*70}")
    print("TRAINING XGBOOST MODEL")
    print(f"{'='*70}")

    print(f"\nHyperparameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  early_stopping_rounds: {early_stopping_rounds}")

    # Initialize model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=early_stopping_rounds,
        verbosity=1
    )

    # Train with early stopping
    print(f"\nTraining...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )

    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Best score: {model.best_score:.4f}")

    return model


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list
) -> dict:
    """
    Comprehensive model evaluation.
    """
    print(f"\n{'='*70}")
    print("MODEL EVALUATION")
    print(f"{'='*70}")

    results = {}

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Training metrics
    results['train'] = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_proba),
    }

    # Test metrics
    results['test'] = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
    }

    # Print metrics
    print(f"\n  {'Metric':<15} {'Train':>10} {'Test':>10} {'Diff':>10}")
    print(f"  {'-'*45}")
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
        train_val = results['train'][metric]
        test_val = results['test'][metric]
        diff = train_val - test_val
        print(f"  {metric:<15} {train_val:>10.4f} {test_val:>10.4f} {diff:>+10.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    results['confusion_matrix'] = cm.tolist()

    print(f"\n  Confusion Matrix (Test):")
    print(f"                    Predicted")
    print(f"                  Absent  Present")
    print(f"  Actual Absent   {cm[0,0]:>6}  {cm[0,1]:>6}")
    print(f"  Actual Present  {cm[1,0]:>6}  {cm[1,1]:>6}")

    # Classification report
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, y_test_pred, target_names=['Absent', 'Present']))

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    results['feature_importance'] = importance.to_dict('records')

    print(f"\n  Top 15 Most Important Features:")
    print(f"  {'Feature':<35} {'Importance':>12}")
    print(f"  {'-'*48}")
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:>12.4f}")

    # Check against target
    target_accuracy = 0.70
    test_accuracy = results['test']['accuracy']

    print(f"\n  {'='*48}")
    if test_accuracy >= target_accuracy:
        print(f"  ✓ TARGET MET: {test_accuracy:.1%} accuracy >= {target_accuracy:.0%} target")
    else:
        print(f"  ✗ TARGET NOT MET: {test_accuracy:.1%} accuracy < {target_accuracy:.0%} target")
        print(f"    Gap: {(target_accuracy - test_accuracy)*100:.1f} percentage points")
    print(f"  {'='*48}")

    return results


def save_model_and_results(
    model,
    results: dict,
    feature_cols: list,
    label_encoders: dict,
    output_dir: Path,
    model_name: str = "xgboost_elk_presence"
):
    """
    Save model, results, and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_file = output_dir / f"{model_name}_{timestamp}.json"
    model.save_model(str(model_file))
    print(f"\n  Model saved to: {model_file}")

    # Also save as pickle for sklearn compatibility
    pickle_file = output_dir / f"{model_name}_{timestamp}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'label_encoders': label_encoders
        }, f)
    print(f"  Pickle saved to: {pickle_file}")

    # Save results
    results_file = output_dir / f"{model_name}_{timestamp}_results.json"

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"  Results saved to: {results_file}")

    # Save feature importance as CSV
    importance_file = output_dir / f"{model_name}_{timestamp}_feature_importance.csv"
    importance_df = pd.DataFrame(results['feature_importance'])
    importance_df.to_csv(importance_file, index=False)
    print(f"  Feature importance saved to: {importance_file}")

    return model_file, results_file


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42
) -> dict:
    """
    Perform stratified k-fold cross-validation.
    """
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION ({n_folds}-fold stratified)")
    print(f"{'='*70}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
        eval_metric='logloss'
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Cross-validation scores
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    cv_roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    print(f"\n  {'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Accuracy':<15} {cv_accuracy.mean():>10.4f} {cv_accuracy.std():>10.4f} {cv_accuracy.min():>10.4f} {cv_accuracy.max():>10.4f}")
    print(f"  {'F1 Score':<15} {cv_f1.mean():>10.4f} {cv_f1.std():>10.4f} {cv_f1.min():>10.4f} {cv_f1.max():>10.4f}")
    print(f"  {'ROC AUC':<15} {cv_roc_auc.mean():>10.4f} {cv_roc_auc.std():>10.4f} {cv_roc_auc.min():>10.4f} {cv_roc_auc.max():>10.4f}")

    return {
        'accuracy': {'mean': cv_accuracy.mean(), 'std': cv_accuracy.std(), 'scores': cv_accuracy.tolist()},
        'f1': {'mean': cv_f1.mean(), 'std': cv_f1.std(), 'scores': cv_f1.tolist()},
        'roc_auc': {'mean': cv_roc_auc.mean(), 'std': cv_roc_auc.std(), 'scores': cv_roc_auc.tolist()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for elk presence prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path('data/features/optimized/complete_context_optimized.csv'),
        help='Input feature file (default: data/features/optimized/complete_context_optimized.csv)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('models'),
        help='Output directory for model and results (default: models)'
    )
    parser.add_argument(
        '--test-years',
        type=int,
        nargs='+',
        default=None,
        help='Years to use for test set (temporal split only, default: last 2 years)'
    )
    parser.add_argument(
        '--split-type',
        type=str,
        choices=['temporal', 'stratified'],
        default='temporal',
        help='Split type: temporal (by year) or stratified (random) (default: temporal)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size for stratified split (default: 0.2)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=500,
        help='Maximum number of boosting rounds (default: 500)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum tree depth (default: 6)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--cross-validate',
        action='store_true',
        help='Also perform cross-validation'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    if not XGBOOST_AVAILABLE:
        print("Error: XGBoost is required. Install with: pip install xgboost")
        return 1

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    print(f"{'='*70}")
    print("PATHWILD XGBoost MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Started at: {datetime.now()}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")

    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_cols, label_encoders = load_and_prepare_data(
        args.input,
        test_years=args.test_years,
        test_size=args.test_size,
        split_type=args.split_type,
        random_state=args.random_state
    )

    # Optional cross-validation
    cv_results = None
    if args.cross_validate:
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        cv_results = cross_validate(X_full, y_full, random_state=args.random_state)

    # Train model
    model = train_xgboost(
        X_train, y_train, X_test, y_test, feature_cols,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state
    )

    # Evaluate
    results = evaluate_model(model, X_train, y_train, X_test, y_test, feature_cols)

    if cv_results:
        results['cross_validation'] = cv_results

    # Save
    save_model_and_results(
        model, results, feature_cols, label_encoders,
        args.output_dir
    )

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Finished at: {datetime.now()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
