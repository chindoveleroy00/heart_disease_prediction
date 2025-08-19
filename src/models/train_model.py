import os
import pandas as pd
import numpy as np
import pickle
import time
import warnings
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# Suppress all warnings (including XGBoost's label encoder warning)
warnings.filterwarnings('ignore')

# Define paths
base_dir = r"C:\Users\lchin\Desktop\Projects\Prototypes\heart_disease_prediction"
featured_data_path = os.path.join(base_dir, 'data', 'processed', 'featured_heart_data.csv')
model_dir = os.path.join(base_dir, 'data', 'models')
model_path = os.path.join(model_dir, 'heart_disease_model.joblib')


def load_featured_data():
    print(f"Loading featured data from {featured_data_path}...")
    df = pd.read_csv(featured_data_path)
    print("Checking data quality...")

    infinities = df.replace([np.inf, -np.inf], np.nan).isna().sum() - df.isna().sum()
    nulls = df.isna().sum()

    if infinities.sum() > 0:
        print(f"Found {infinities.sum()} infinite values across {sum(infinities > 0)} columns")
    if nulls.sum() > 0:
        print(f"Found {nulls.sum()} null values across {sum(nulls > 0)} columns")

    df = df.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy='median')
    cols_with_missing = df.columns[df.isna().any()].tolist()
    if cols_with_missing:
        print(f"Imputing missing values in {len(cols_with_missing)} columns using median")
        df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])

    return df


def train_test_split_data(df, target_col='heart_disease', test_size=0.2, random_state=42):
    print(f"Splitting data into training and testing sets (test size: {test_size})...")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        X = X.drop(constant_cols, axis=1)

    for col in X.columns:
        max_val = X[col].max()
        min_val = X[col].min()
        if max_val > 1e10 or min_val < -1e10:
            print(f"Column '{col}' has extreme values: min={min_val}, max={max_val}. Clipping values.")
            X[col] = X[col].clip(-1e10, 1e10)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_xgboost_model(X_train, y_train):
    print("\nTraining XGBoost model...")
    start_time = time.time()

    model = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        missing=np.nan
    )

    try:
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"  Training time: {training_time:.2f} seconds")
        return model, training_time
    except Exception as e:
        print(f"Error during XGBoost training: {str(e)}")
        print("\nAttempting to diagnose and fix the issue...")
        problematic_cols = [col for col in X_train.columns if X_train[col].isna().any() or np.isinf(X_train[col]).any()]

        if problematic_cols:
            print(f"Found {len(problematic_cols)} problematic columns with NaN or infinite values")
            if len(problematic_cols) < len(X_train.columns) * 0.2:
                print(f"Dropping problematic columns and retraining: {problematic_cols}")
                X_train_clean = X_train.drop(problematic_cols, axis=1)
                model.fit(X_train_clean, y_train)
                training_time = time.time() - start_time
                model.dropped_columns = problematic_cols
                return model, training_time
            else:
                print("Too many problematic columns. Using median imputation instead of dropping.")
                imputer = SimpleImputer(strategy='median')
                X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
                model.fit(X_train_imputed, y_train)
                model.imputer = imputer
                training_time = time.time() - start_time
                return model, training_time
        else:
            raise


def evaluate_model(model, X_test, y_test):
    print("\nModel Evaluation Results:")
    print("-" * 60)

    if hasattr(model, 'dropped_columns'):
        print(f"Note: Model was trained without {len(model.dropped_columns)} problematic columns")
        X_test = X_test.drop(model.dropped_columns, axis=1)
    elif hasattr(model, 'imputer'):
        print("Note: Using imputer to handle missing values in test data")
        X_test = pd.DataFrame(model.imputer.transform(X_test), columns=X_test.columns)

    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        for col in X_test.columns:
            if X_test[col].isna().any():
                X_test[col] = X_test[col].fillna(X_test[col].median())
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

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


def fine_tune_xgboost(model, X_train, y_train, X_test, y_test):
    print("\nFine-tuning XGBoost model...")

    if hasattr(model, 'dropped_columns'):
        X_train = X_train.drop(model.dropped_columns, axis=1)
        X_test = X_test.drop(model.dropped_columns, axis=1)
    elif hasattr(model, 'imputer'):
        X_train = pd.DataFrame(model.imputer.transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(model.imputer.transform(X_test), columns=X_test.columns)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    print(f"Grid search will evaluate {np.prod([len(v) for v in param_grid.values()])} combinations")
    grid_search = GridSearchCV(
        estimator=XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',
            missing=np.nan
        ),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    print("Running grid search (this may take some time)...")
    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        if hasattr(model, 'dropped_columns'):
            best_model.dropped_columns = model.dropped_columns
        if hasattr(model, 'imputer'):
            best_model.imputer = model.imputer

        print("\nEvaluating fine-tuned model:")
        results = evaluate_model(best_model, X_test, y_test)
        return best_model, results
    except Exception as e:
        print(f"Error during grid search: {str(e)}")
        print("Skipping fine-tuning and returning original model")
        return model, evaluate_model(model, X_test, y_test)


def analyze_feature_importance(model, X_train):
    print("\nFeature Importance Analysis:")
    if hasattr(model, 'dropped_columns'):
        X_train = X_train.drop(model.dropped_columns, axis=1)
    elif hasattr(model, 'imputer'):
        X_train = pd.DataFrame(model.imputer.transform(X_train), columns=X_train.columns)

    try:
        importance = model.feature_importances_
        df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
        df = df.sort_values('Importance', ascending=False)
        print(df.head(20))
        return df
    except Exception as e:
        print(f"Error analyzing feature importance: {str(e)}")
        return None


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"\nSaving model to {path}...")
    try:
        # Use protocol=4 for better compatibility
        joblib.dump(model, path, protocol=4, compress=('zlib', 3))
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

def main():
    try:
        df = load_featured_data()
        X_train, X_test, y_train, y_test = train_test_split_data(df)
        initial_model, _ = train_xgboost_model(X_train, y_train)
        evaluate_model(initial_model, X_test, y_test)
        fine_tuned_model, _ = fine_tune_xgboost(initial_model, X_train, y_train, X_test, y_test)
        analyze_feature_importance(fine_tuned_model, X_train)
        save_model(fine_tuned_model, model_path)
        print("\nModel training and evaluation complete!")
    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
