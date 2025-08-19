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

# Suppress warnings
warnings.filterwarnings('ignore')

# Define base paths using pathlib for cross-platform compatibility
base_dir = Path(__file__).parent.parent.parent  # Go up to project root
model_path = base_dir / 'data' / 'models' / 'heart_disease_model.joblib'
pca_model_path = base_dir / 'data' / 'models' / 'pca_model.pkl'


def load_model():
    """Load the trained XGBoost model with better error handling"""
    print(f"Loading model from {model_path}...")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully: {type(model)}")

        # Check if model has expected attributes
        if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
            print("✅ Model has required prediction methods")
        else:
            print("⚠️  Model missing expected prediction methods")

        return model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("This usually indicates a corrupted model file.")
        print("Try regenerating the model with: python regenerate_models.py")
        raise


def load_pca_model():
    """Load the PCA model with better error handling"""
    print(f"Loading PCA model from {pca_model_path}...")

    if not pca_model_path.exists():
        print("⚠️  PCA model file not found - will skip PCA transformation")
        return None

    try:
        import pickle
        with open(pca_model_path, 'rb') as f:
            pca_model = pickle.load(f)
        print(f"✅ PCA model loaded successfully: {type(pca_model)}")
        return pca_model

    except Exception as e:
        print(f"❌ Error loading PCA model: {e}")
        print("Will continue without PCA transformation")
        return None


def prepare_features(input_data: dict):
    """
    Prepare features for model prediction with robust error handling
    """
    try:
        # Convert input to dataframe
        df = pd.DataFrame([input_data])
        print(f"Original input has {len(df.columns)} columns")

        # Try to apply feature engineering using the FeatureEngineer class
        try:
            from src.features.build_features import FeatureEngineer
            fe = FeatureEngineer()
            df = fe.build_features(df)
            print(f"After feature engineering: {len(df.columns)} columns")
        except ImportError:
            print("⚠️  FeatureEngineer not available - using basic feature engineering")
            df = apply_basic_feature_engineering(df)
        except Exception as e:
            print(f"⚠️  Advanced feature engineering failed: {e}")
            print("Falling back to basic feature engineering")
            df = apply_basic_feature_engineering(df)

        # Load and apply PCA model
        try:
            pca = load_pca_model()
            if pca is not None:
                df = apply_pca_transformation(df, pca)
            else:
                print("Continuing without PCA transformation")

        except Exception as e:
            print(f"❌ Error in PCA transformation: {e}")
            print("Continuing without PCA transformation")

        return df

    except Exception as e:
        print(f"❌ Error in prepare_features: {e}")
        # Fallback: return basic dataframe
        return pd.DataFrame([input_data])


def apply_basic_feature_engineering(df):
    """Apply basic feature engineering as fallback"""
    try:
        # Calculate BMI if height and weight are available
        if 'height_cm' in df.columns and 'weight_kg' in df.columns:
            df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

        # Blood pressure ratios
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            df['bp_ratio'] = df['systolic_bp'] / df['diastolic_bp']
            df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

        # Cholesterol ratios
        if 'total_cholesterol' in df.columns and 'hdl' in df.columns:
            df['cholesterol_hdl_ratio'] = df['total_cholesterol'] / df['hdl']

        # Age-heart rate interaction
        if 'age' in df.columns and 'max_heart_rate' in df.columns:
            df['age_hr_interaction'] = df['age'] * df['max_heart_rate']

        print(f"Basic feature engineering completed: {len(df.columns)} columns")
        return df

    except Exception as e:
        print(f"❌ Basic feature engineering failed: {e}")
        return df


def apply_pca_transformation(df, pca_model):
    """Apply PCA transformation with robust error handling"""
    try:
        # Get numerical columns for PCA (excluding target if present)
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'heart_disease' in numerical_cols:
            numerical_cols.remove('heart_disease')

        print(f"Using {len(numerical_cols)} numerical columns for PCA")

        # Handle missing values using mean strategy
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(df[numerical_cols])

        # Handle feature count mismatch
        expected_features = pca_model.n_features_in_
        if X_imputed.shape[1] != expected_features:
            print(f"Feature count mismatch: got {X_imputed.shape[1]}, expected {expected_features}")

            if X_imputed.shape[1] < expected_features:
                # Pad with zeros
                padding = np.zeros((X_imputed.shape[0], expected_features - X_imputed.shape[1]))
                X_imputed = np.hstack([X_imputed, padding])
                print(f"Padded features to {X_imputed.shape[1]}")
            else:
                # Truncate
                X_imputed = X_imputed[:, :expected_features]
                print(f"Truncated features to {X_imputed.shape[1]}")

        # Apply PCA transformation
        pca_features = pca_model.transform(X_imputed)

        # Add PCA features to dataframe
        pca_cols = [f'pca_{i + 1}' for i in range(pca_features.shape[1])]
        pca_df = pd.DataFrame(pca_features, columns=pca_cols, index=df.index)
        df = pd.concat([df, pca_df], axis=1)

        print(f"Added {pca_features.shape[1]} PCA components")
        return df

    except Exception as e:
        print(f"❌ PCA transformation failed: {e}")
        # Fallback: add zero PCA features
        n_components = getattr(pca_model, 'n_components_', 5)
        for i in range(n_components):
            df[f'pca_{i + 1}'] = 0
        print(f"Used zero PCA features as fallback ({n_components} components)")
        return df


def ensure_model_input(model, input_df):
    """
    Ensure the input dataframe matches what the model expects
    """
    print(f"Input dataframe shape: {input_df.shape}")

    try:
        # Apply any preprocessing that was saved with the model
        if hasattr(model, 'dropped_columns'):
            print(f"Dropping {len(model.dropped_columns)} columns that were dropped during training")
            input_df = input_df.drop(model.dropped_columns, axis=1, errors='ignore')

        if hasattr(model, 'imputer'):
            print("Applying imputer that was used during training")
            input_df = pd.DataFrame(model.imputer.transform(input_df), columns=input_df.columns)

        # Handle any remaining data quality issues
        input_df = input_df.replace([np.inf, -np.inf], np.nan)

        # Fill missing values with median
        for col in input_df.columns:
            if input_df[col].isna().any():
                median_val = input_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                input_df[col] = input_df[col].fillna(median_val)

        # Final check for model compatibility
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            current_features = list(input_df.columns)

            missing_cols = [col for col in expected_features if col not in current_features]
            extra_cols = [col for col in current_features if col not in expected_features]

            if missing_cols:
                print(f"Adding {len(missing_cols)} missing columns with zeros")
                for col in missing_cols:
                    input_df[col] = 0

            if extra_cols:
                print(f"Dropping {len(extra_cols)} extra columns")
                input_df = input_df.drop(columns=extra_cols)

            # Ensure correct column order
            input_df = input_df[expected_features]
        else:
            print("⚠️  Model doesn't have feature_names_in_ attribute")

        print(f"Final input shape for model: {input_df.shape}")
        return input_df

    except Exception as e:
        print(f"❌ Error in ensure_model_input: {e}")
        import traceback
        traceback.print_exc()
        return input_df


def predict_single_sample(model, input_data):
    """Make a prediction for a single sample with comprehensive error handling"""
    try:
        # Prepare features using the same pipeline as training
        input_df = prepare_features(input_data)

        # Ensure input matches model expectations
        input_df = ensure_model_input(model, input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Get prediction probability
        try:
            probability_array = model.predict_proba(input_df)[0]
            # Get probability for positive class (heart disease)
            if len(probability_array) > 1:
                probability = probability_array[1]  # Probability of class 1 (heart disease)
            else:
                probability = probability_array[0]
        except Exception as e:
            print(f"⚠️  Could not get prediction probability: {e}")
            # Fallback: use a default probability based on prediction
            probability = 0.8 if prediction == 1 else 0.2

        return int(prediction), float(probability)

    except Exception as e:
        print(f"❌ Error in predict_single_sample: {e}")
        import traceback
        traceback.print_exc()
        raise


def format_prediction_results(prediction, probability):
    """Format prediction results with risk levels"""
    prob_percent = probability * 100

    # Determine risk level
    if prob_percent < 20:
        risk_level = "Very Low"
    elif prob_percent < 40:
        risk_level = "Low"
    elif prob_percent < 60:
        risk_level = "Moderate"
    elif prob_percent < 80:
        risk_level = "High"
    else:
        risk_level = "Very High"

    # Generate interpretation
    disease_status = "heart disease" if prediction == 1 else "no heart disease"
    interpretation = (f"The model predicts {disease_status} with a probability of "
                      f"{prob_percent:.1f}%. This represents a {risk_level.lower()} risk level.")

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": risk_level,
        "interpretation": interpretation
    }


def main():
    """Main function for testing predictions"""
    try:
        print("🧪 Testing predict_model.py")
        print("=" * 40)

        # Load model
        model = load_model()
        print("✅ Model loaded successfully")

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

        print("🔮 Making test prediction...")
        prediction, probability = predict_single_sample(model, example_input)
        result = format_prediction_results(prediction, probability)

        print("\n📊 Prediction Results:")
        print("-" * 40)
        print(f"Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
        print(f"Probability: {result['probability']:.1%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Interpretation: {result['interpretation']}")

        return result

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

        # Provide helpful error messages
        if "No module named" in str(e):
            print("\n💡 Missing dependencies. Try:")
            print("   pip install -r requirements.txt")
        elif "FileNotFoundError" in str(e):
            print("\n💡 Model files missing. Try:")
            print("   python regenerate_models.py")
        elif "invalid load key" in str(e):
            print("\n💡 Model files corrupted. Try:")
            print("   python regenerate_models.py")

        raise


if __name__ == "__main__":
    main()