import os
import pandas as pd
import numpy as np
import joblib
import warnings

# Import Tuple for type hinting
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from pathlib import Path

# Suppress warnings (e.g., from XGBoost's label encoder warning, etc.)
warnings.filterwarnings('ignore')

# Define base paths using pathlib for cross-platform compatibility
base_dir = Path(r"C:\Users\lchin\Desktop\Projects\Prototypes\heart_disease_prediction")
featured_data_path = base_dir / 'data' / 'processed' / 'featured_heart_data.csv'
model_path = base_dir / 'data' / 'models' / 'heart_disease_model.joblib'


def load_featured_data(file_path: Path) -> pd.DataFrame:
    """
    Loads the featured data and performs initial data quality checks and imputation,
    mirroring the logic in train_model.py.
    """
    print(f"Loading featured data from {file_path}...")
    df = pd.read_csv(file_path)
    print("Checking data quality...")

    infinities = df.replace([np.inf, -np.inf], np.nan).isna().sum() - df.isna().sum()
    nulls = df.isna().sum()

    if infinities.sum() > 0:
        print(f"Found {infinities.sum()} infinite values across {sum(infinities > 0)} columns")
    if nulls.sum() > 0:
        print(f"Found {nulls.sum()} null values across {sum(nulls > 0)} columns")

    # Replace infinities with NaN for imputation
    df = df.replace([np.inf, -np.inf], np.nan)

    # Impute missing values using median strategy (matching train_model.py)
    imputer = SimpleImputer(strategy='median')
    cols_with_missing = df.columns[df.isna().any()].tolist()
    if cols_with_missing:
        print(f"Imputing missing values in {len(cols_with_missing)} columns using median")
        df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])

    return df


def train_test_split_data(df: pd.DataFrame, target_col: str = 'heart_disease', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets, exactly mirroring train_model.py logic.
    """
    print(f"Splitting data into training and testing sets (test size: {test_size})...")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Drop constant columns - must match logic in train_model.py
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        X = X.drop(constant_cols, axis=1)

    # Handle extreme values by clipping - must match logic in train_model.py
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            max_val = X[col].max()
            min_val = X[col].min()
            if max_val > 1e10 or min_val < -1e10:
                print(f"Column '{col}' has extreme values: min={min_val}, max={max_val}. Clipping values.")
                X[col] = X[col].clip(-1e10, 1e10)

    # Use stratified split if possible
    if y.nunique() > 1:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        print("Warning: Target variable has only one unique value, skipping stratification.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_model(model_path: Path):
    """Loads the trained model from the specified path."""
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the trained model on the test data, exactly matching train_model.py logic.
    """
    print("\nModel Evaluation Results:")
    print("-" * 60)

    # Apply same preprocessing as in train_model.py
    X_test_processed = X_test.copy()
    
    if hasattr(model, 'dropped_columns'):
        print(f"Note: Model was trained without {len(model.dropped_columns)} problematic columns")
        X_test_processed = X_test_processed.drop(model.dropped_columns, axis=1, errors='ignore')
    elif hasattr(model, 'imputer'):
        print("Note: Using imputer to handle missing values in test data")
        X_test_processed = pd.DataFrame(model.imputer.transform(X_test_processed), columns=X_test_processed.columns)

    # Handle any remaining issues exactly as in train_model.py
    try:
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    except Exception:
        print("Handling prediction errors with fallback imputation...")
        X_test_processed = X_test_processed.replace([np.inf, -np.inf], np.nan)
        for col in X_test_processed.columns:
            if X_test_processed[col].isna().any():
                X_test_processed[col] = X_test_processed[col].fillna(X_test_processed[col].median())
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"{'Metric':<15} {'Value':<10}")
    print("-" * 60)
    print(f"{'Accuracy':<15} {accuracy:.4f}")
    print(f"{'Precision':<15} {precision:.4f}")
    print(f"{'Recall':<15} {recall:.4f}")
    print(f"{'F1 Score':<15} {f1:.4f}")
    print(f"{'ROC AUC':<15} {roc_auc:.4f}")
    print("-" * 60)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }


def main():
    try:
        # Load the featured data
        df = load_featured_data(featured_data_path)

        # Split data exactly as in train_model.py (CRITICAL: same random_state=42)
        X_train, X_test, y_train, y_test = train_test_split_data(df)

        # Load the trained model
        model = load_model(model_path)

        # Evaluate the model on the test set
        evaluation_results = evaluate_model(model, X_test, y_test)

        print("\nModel evaluation complete!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the featured data and model files exist at the specified paths.")
    except ValueError as e:
        print(f"Data or Model Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()