import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
import joblib
import json
from typing import Tuple, Dict, Any


class FeatureEngineer:
    """Streamlined feature engineering for heart disease prediction"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.feature_config = {
            'core_features': [
                'age', 'sex', 'systolic_bp', 'diastolic_bp', 'resting_heart_rate',
                'total_cholesterol', 'hdl', 'ldl', 'fasting_glucose', 'bmi',
                'smoking_status', 'exercise_induced_angina', 'st_depression'
            ],
            'derived_features': [
                'age_decile', 'chol_ratio', 'bmi_category', 'bp_category',
                'mean_arterial_pressure', 'rate_pressure_product'
            ]
        }

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load processed data with validation"""
        try:
            print(f"Loading data from {file_path}...")
            df = pd.read_csv(file_path)
            required_cols = ['age', 'sex', 'systolic_bp', 'diastolic_bp', 
                           'total_cholesterol', 'hdl', 'heart_disease']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Missing required columns in input data")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def create_core_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create essential clinical features"""
        # Age features
        df['age_decile'] = pd.qcut(df['age'], q=10, labels=False, duplicates='drop')
        
        # Blood pressure features
        df['mean_arterial_pressure'] = df['diastolic_bp'] + 0.33 * (df['systolic_bp'] - df['diastolic_bp'])
        df['rate_pressure_product'] = df['systolic_bp'] * df['resting_heart_rate']
        
        # Cholesterol features
        df['chol_ratio'] = df['total_cholesterol'] / df['hdl']
        
        # BMI categories
        df['bmi_category'] = pd.cut(df['bmi'],
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=[0, 1, 2, 3])
        
        # BP categories
        df['bp_category'] = pd.cut(df['systolic_bp'],
                                  bins=[0, 120, 130, 140, 180, 200],
                                  labels=[0, 1, 2, 3, 4])
        
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create clinically relevant interaction terms"""
        # Age-cholesterol interaction
        df['age_chol_interaction'] = df['age'] * (df['total_cholesterol'] / 100)
        
        # BMI-heart rate interaction
        df['bmi_hr_interaction'] = df['bmi'] * (df['resting_heart_rate'] / 100)
        
        return df

    def create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk scores"""
        # Simple risk score
        risk_factors = [
            ('age', 0.05),
            ('sex', 0.5),  # Male=1, Female=0
            ('systolic_bp', 0.01),
            ('total_cholesterol', 0.005),
            ('smoking_status', 0.3),
            ('exercise_induced_angina', 0.4)
        ]
        
        df['simple_risk_score'] = 0
        for col, weight in risk_factors:
            if col in df.columns:
                df['simple_risk_score'] += df[col] * weight
        
        return df

    def save_config(self):
        """Save feature engineering configuration"""
        if self.config_path:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.feature_config, f, indent=2)
            print(f"Feature configuration saved to {self.config_path}")

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Orchestrate all feature engineering steps"""
        df = self.create_core_features(df)
        df = self.create_interaction_features(df)
        df = self.create_risk_scores(df)
        self.save_config()
        return df


def create_pca_features(df: pd.DataFrame, target_col: str = 'heart_disease',
                       model_path: str = None, n_components: int = 5) -> Tuple[pd.DataFrame, PCA]:
    """
    Create PCA features from numerical columns
    
    Args:
        df: Input dataframe (after feature engineering)
        target_col: Name of target column
        model_path: Path to save PCA model
        n_components: Number of PCA components to keep
        
    Returns:
        Tuple of (dataframe with PCA features, PCA model)
    """
    # Select numerical features excluding target
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df[numerical_cols])
    
    # Run PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(X_imputed)
    
    # Add PCA features to dataframe
    pca_cols = [f'pca_{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(pca_features, columns=pca_cols, index=df.index)
    df = pd.concat([df, df_pca], axis=1)
    
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pca, model_path)
        print(f"PCA model saved to {model_path}")
    
    print(f"Created {n_components} PCA components explaining "
          f"{sum(pca.explained_variance_ratio_):.2%} variance")
    
    return df, pca


def save_dataset(df: pd.DataFrame, save_path: str):
    """Save dataset with validation"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset saved to {save_path} with shape {df.shape}")


def main():
    # Configuration
    BASE_DIR = r"C:\Users\lchin\Desktop\Projects\Prototypes\heart_disease_prediction"
    PATHS = {
        'processed': os.path.join(BASE_DIR, 'data', 'processed', 'processed_heart_data.csv'),
        'featured': os.path.join(BASE_DIR, 'data', 'processed', 'featured_heart_data.csv'),
        'config': os.path.join(BASE_DIR, 'src', 'features', 'feature_config.json'),
        'pca_model': os.path.join(BASE_DIR, 'data', 'models', 'pca_model.pkl')
    }

    try:
        # Initialize feature engineer
        fe = FeatureEngineer(config_path=PATHS['config'])

        # Load and build features
        df = fe.load_data(PATHS['processed'])
        df = fe.build_features(df)

        # Add PCA features
        df, pca_model = create_pca_features(
            df,
            target_col='heart_disease',
            model_path=PATHS['pca_model'],
            n_components=5
        )

        # Save results
        save_dataset(df, PATHS['featured'])

        # Final report
        print("\nFeature Engineering Report:")
        print(f"- Original features: {len(fe.feature_config['core_features'])}")
        print(f"- Derived features: {len(fe.feature_config['derived_features'])}")
        print(f"- PCA components added: 5")
        print(f"- Total features created: {len(df.columns)}")
        print(f"- Total variance explained by PCA: {sum(pca_model.explained_variance_ratio_):.2%}")

    except Exception as e:
        print(f"\nError in feature engineering pipeline: {e}")
        raise


if __name__ == "__main__":
    main()