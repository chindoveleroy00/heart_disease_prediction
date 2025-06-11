import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Add the project root to Python path to enable absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import from the features module using absolute path
from src.features.build_features import FeatureEngineer, create_pca_features

# Suppress warnings
warnings.filterwarnings('ignore')

# Define base paths using pathlib for cross-platform compatibility
base_dir = Path(r"C:\Users\lchin\Desktop\Projects\Prototypes\heart_disease_prediction")
model_path = base_dir / 'data' / 'models' / 'heart_disease_model.joblib'
pca_model_path = base_dir / 'data' / 'models' / 'pca_model.pkl'


def load_model():
    """Load the trained XGBoost model"""
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def load_pca_model():
    """Load the PCA model"""
    print(f"Loading PCA model from {pca_model_path}...")
    if not pca_model_path.exists():
        raise FileNotFoundError(f"PCA model file not found at {pca_model_path}")
    return joblib.load(pca_model_path)


def prepare_features(input_data: dict):
    """
    Prepare features for model prediction following the exact same pipeline as build_features.py
    """
    # Convert input to dataframe
    df = pd.DataFrame([input_data])
    print(f"Original input has {len(df.columns)} columns")

    # Apply feature engineering using the same FeatureEngineer class
    fe = FeatureEngineer()
    df = fe.build_features(df)
    print(f"After feature engineering: {len(df.columns)} columns")

    # Load PCA model and apply transformation
    try:
        pca = load_pca_model()
        
        # Get numerical columns for PCA (excluding target if present)
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'heart_disease' in numerical_cols:
            numerical_cols.remove('heart_disease')

        print(f"Using {len(numerical_cols)} numerical columns for PCA")

        # Handle missing values exactly as in build_features.py (using mean strategy)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(df[numerical_cols])
        
        # Apply PCA transformation
        pca_features = pca.transform(X_imputed)
        
        # Add PCA features to dataframe
        pca_cols = [f'pca_{i+1}' for i in range(pca.n_components_)]
        pca_df = pd.DataFrame(pca_features, columns=pca_cols, index=df.index)
        df = pd.concat([df, pca_df], axis=1)
        
        print(f"Added {pca.n_components_} PCA components")
        
    except Exception as e:
        print(f"Error in PCA transformation: {e}")
        # Fallback: add zero PCA features
        for i in range(5):  # Default 5 components as in build_features.py
            df[f'pca_{i+1}'] = 0
        print("Used zero PCA features as fallback")

    return df


def ensure_model_input(model, input_df):
    """
    Ensure the input dataframe matches what the model expects,
    following the same preprocessing as train_model.py
    """
    print(f"Input dataframe shape: {input_df.shape}")
    
    # Apply the same preprocessing that might have been done during training
    if hasattr(model, 'dropped_columns'):
        print(f"Dropping {len(model.dropped_columns)} columns that were dropped during training")
        input_df = input_df.drop(model.dropped_columns, axis=1, errors='ignore')
    elif hasattr(model, 'imputer'):
        print("Applying imputer that was used during training")
        input_df = pd.DataFrame(model.imputer.transform(input_df), columns=input_df.columns)

    # Handle any remaining data quality issues exactly as in train_model.py
    input_df = input_df.replace([np.inf, -np.inf], np.nan)
    for col in input_df.columns:
        if input_df[col].isna().any():
            input_df[col] = input_df[col].fillna(input_df[col].median())

    # Final check for model compatibility
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
        missing_cols = [col for col in expected_features if col not in input_df.columns]
        extra_cols = [col for col in input_df.columns if col not in expected_features]

        if missing_cols:
            print(f"Adding {len(missing_cols)} missing columns with zeros")
            for col in missing_cols:
                input_df[col] = 0

        if extra_cols:
            print(f"Dropping {len(extra_cols)} extra columns")
            input_df = input_df.drop(columns=extra_cols)

        # Ensure correct column order
        input_df = input_df[expected_features]

    print(f"Final input shape for model: {input_df.shape}")
    
    return input_df


def predict_single_sample(model, input_data):
    """Make a prediction for a single sample"""
    # Prepare features using the same pipeline as training
    input_df = prepare_features(input_data)
    
    # Ensure input matches model expectations
    input_df = ensure_model_input(model, input_df)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]
    
    return prediction, probability


def format_prediction_results(prediction, probability):
    """Format prediction results with risk levels"""
    if probability < 0.2:
        risk_level = "Very Low"
    elif probability < 0.4:
        risk_level = "Low"
    elif probability < 0.6:
        risk_level = "Moderate"
    elif probability < 0.8:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": risk_level,
        "interpretation": f"The model predicts {'heart disease' if prediction == 1 else 'no heart disease'} "
                          f"with a probability of {probability:.2%}. This represents a {risk_level} risk level."
    }


def main():
    """Main function for testing predictions"""
    try:
        model = load_model()

        # Example input with all required base features
        example_input = {
            'age': 60,
            'sex': 1,
            'ethnicity': 0,
            'family_history': 1,
            'previous_cardiac_events': 0,
            'diabetes_status': 1,
            'hypertension': 1,
            'systolic_bp': 145,
            'diastolic_bp': 95,
            'resting_heart_rate': 82,
            'height_cm': 170,
            'weight_kg': 85,
            'bmi': 29.4,
            'total_cholesterol': 240,
            'hdl': 35,
            'ldl': 180,
            'fasting_glucose': 110,
            'hba1c': 6.2,
            'smoking_status': 1,
            'alcohol_per_week': 5,
            'physical_activity_hours': 1,
            'diet_quality': 2,
            'stress_level': 4,
            'chest_pain_type': 1,
            'exercise_induced_angina': 1,
            'resting_ecg': 1,
            'max_heart_rate': 155,
            'st_depression': 0.8,
            'st_slope': 1,
            'num_vessels': 1,
            'thalassemia': 1
        }

        prediction, probability = predict_single_sample(model, example_input)
        result = format_prediction_results(prediction, probability)

        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
        print(f"Probability: {result['probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Interpretation: {result['interpretation']}")
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()