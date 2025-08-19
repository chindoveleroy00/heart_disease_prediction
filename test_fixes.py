# test_fixes.py - Test script to verify the fixes work

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def test_flask_imports():
    """Test if Flask app can be imported without errors"""
    try:
        print("Testing Flask app imports...")

        # Test importing the app factory
        from app import create_app
        print("✅ App factory imported successfully")

        # Test importing models
        from app.models import Prediction, PredictionSummary, ModelVersion
        print("✅ Models imported successfully")

        # Test importing forms
        from app.forms import PredictionForm
        print("✅ Forms imported successfully")

        return True

    except Exception as e:
        print(f"❌ Error testing Flask imports: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test if the model can be loaded successfully"""
    try:
        from app.routes import load_model

        print("Testing model loading...")
        model_dict = load_model()

        if model_dict is None:
            print("❌ Model loading failed")
            return False

        if 'model' not in model_dict:
            print("❌ Model dictionary missing 'model' key")
            return False

        print("✅ Model loaded successfully")
        print(f"Model type: {type(model_dict['model'])}")
        print(f"PCA available: {'pca' in model_dict and model_dict['pca'] is not None}")

        return True

    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_pipeline():
    """Test the prediction pipeline with sample data"""
    try:
        from app.routes import load_model, predict_single_sample, format_prediction_results

        print("Testing prediction pipeline...")

        # Load model
        model_dict = load_model()
        if model_dict is None:
            print("❌ Cannot test prediction - model not loaded")
            return False

        # Sample input data
        sample_data = {
            'age': 45,
            'sex': 1,  # Male
            'ethnicity': 0,
            'family_history': 0,
            'previous_cardiac_events': 0,
            'diabetes_status': 0,
            'hypertension': 1,
            'systolic_bp': 140,
            'diastolic_bp': 90,
            'resting_heart_rate': 75,
            'height_cm': 175,
            'weight_kg': 80,
            'bmi': 26.1,
            'total_cholesterol': 240,
            'hdl': 40,
            'ldl': 160,
            'fasting_glucose': 100,
            'hba1c': 5.8,
            'smoking_status': 1,  # Former
            'alcohol_per_week': 2,
            'physical_activity_hours': 3,
            'diet_quality': 2,
            'stress_level': 3,
            'chest_pain_type': 1,
            'exercise_induced_angina': 0,
            'resting_ecg': 0,
            'max_heart_rate': 150,
            'st_depression': 1.0,
            'st_slope': 1,
            'num_vessels': 1,
            'thalassemia': 2
        }

        # Make prediction
        prediction, probability = predict_single_sample(model_dict, sample_data)
        result = format_prediction_results(prediction, probability)

        print("✅ Prediction pipeline successful")
        print(f"Prediction: {prediction}")
        print(f"Probability: {probability:.2%}")
        print(f"Risk Level: {result['risk_level']}")

        return True

    except Exception as e:
        print(f"❌ Error testing prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_connection():
    """Test database connection"""
    try:
        print("Testing database connection...")

        # Check if database file exists
        db_path = project_root / "database" / "heart_disease.db"

        if db_path.exists():
            print("✅ Database file exists")
            print(f"Database path: {db_path}")
        else:
            print("⚠️  Database file not found - will be created on first run")

        return True

    except Exception as e:
        print(f"❌ Error testing database: {e}")
        return False


def test_model_files_exist():
    """Test if model files exist and are readable"""
    try:
        print("Testing model files...")

        model_path = project_root / "data" / "models" / "heart_disease_model.joblib"
        pca_path = project_root / "data" / "models" / "pca_model.pkl"

        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False

        if not pca_path.exists():
            print(f"⚠️  PCA model file not found: {pca_path}")
            print("   This is optional but recommended")

        # Try to check file size
        model_size = model_path.stat().st_size
        print(f"✅ Model file exists ({model_size:,} bytes)")

        if pca_path.exists():
            pca_size = pca_path.stat().st_size
            print(f"✅ PCA model file exists ({pca_size:,} bytes)")

        return True

    except Exception as e:
        print(f"❌ Error checking model files: {e}")
        return False


def regenerate_models_if_needed():
    """Try to regenerate models if they're missing or corrupted"""
    try:
        print("Checking if models need regeneration...")

        model_path = project_root / "data" / "models" / "heart_disease_model.joblib"
        featured_data_path = project_root / "data" / "processed" / "featured_heart_data.csv"

        if not model_path.exists():
            print("Model file missing. Checking if training data exists...")

            if featured_data_path.exists():
                print("Training data found. You can regenerate models by running:")
                print("  python src/models/train_model.py")
                return False
            else:
                print("❌ Both model and training data missing!")
                print("You need to run the full data processing pipeline:")
                print("  1. python src/data/preprocess.py")
                print("  2. python src/features/build_features.py")
                print("  3. python src/models/train_model.py")
                return False

        return True

    except Exception as e:
        print(f"❌ Error checking model regeneration: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Running Heart Disease Prediction App Tests")
    print("=" * 50)

    tests = [
        ("Flask Imports", test_flask_imports),
        ("Model Files Check", test_model_files_exist),
        ("Model Loading", test_model_loading),
        ("Prediction Pipeline", test_prediction_pipeline),
        ("Database Connection", test_database_connection),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))

    print("\n🏁 Test Summary")
    print("=" * 50)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1

    print(f"\nTests passed: {passed}/{len(results)}")

    if passed == len(results):
        print("🎉 All tests passed! Your app should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

        # Check if we need to regenerate models
        if not regenerate_models_if_needed():
            print("\n💡 Suggested fix: Regenerate the machine learning models")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)