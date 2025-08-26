#!/usr/bin/env python3
"""
Script to retrain and fix the PCA model
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

def retrain_pca_model():
    """Retrain the PCA model with proper feature engineering"""
    
    # Paths
    base_dir = Path(__file__).parent.parent
    featured_data_path = base_dir / "data" / "processed" / "featured_heart_data.csv"
    pca_model_path = base_dir / "data" / "models" / "pca_model.pkl"
    
    print("🔄 Retraining PCA model...")
    
    # Load featured data
    if not featured_data_path.exists():
        print(f"❌ Featured data not found at {featured_data_path}")
        return False
    
    df = pd.read_csv(featured_data_path)
    print(f"📊 Loaded data with shape: {df.shape}")
    
    # Separate features and target
    if 'heart_disease' not in df.columns:
        print("❌ Target column 'heart_disease' not found")
        return False
    
    X = df.drop('heart_disease', axis=1)
    print(f"📋 Features: {list(X.columns)}")
    print(f"📊 Feature count: {X.shape[1]}")
    
    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create new PCA model
    pca = PCA(n_components=0.95, random_state=42)  # Keep 95% variance
    pca.fit(X_scaled)
    
    # Save the PCA model
    os.makedirs(os.path.dirname(pca_model_path), exist_ok=True)
    joblib.dump(pca, pca_model_path)
    
    print(f"✅ PCA model retrained and saved with {pca.n_components_} components")
    print(f"📊 Original features: {X.shape[1]}")
    print(f"📊 PCA components: {pca.n_components_}")
    print(f"📊 Variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    return True

if __name__ == "__main__":
    success = retrain_pca_model()
    if success:
        print("🎉 PCA model retraining completed successfully!")
    else:
        print("❌ PCA model retraining failed!")