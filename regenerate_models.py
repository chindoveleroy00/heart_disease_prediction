#!/usr/bin/env python3
"""
regenerate_models.py - Script to regenerate corrupted or missing models

This script will:
1. Check if training data exists
2. Regenerate models using the existing training pipeline
3. Verify the new models work correctly
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def check_training_data():
    """Check if required training data exists"""
    print("🔍 Checking training data availability...")

    required_files = [
        "data/processed/featured_heart_data.csv",
        "data/raw/zimbabwe_heart_disease_data.csv"
    ]

    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            size = full_path.stat().st_size
            print(f"✅ Found {file_path} ({size:,} bytes)")

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False, missing_files

    return True, []


def backup_existing_models():
    """Backup existing models before regenerating"""
    print("💾 Backing up existing models...")

    model_dir = project_root / "data" / "models"
    backup_dir = model_dir / "backup"
    backup_dir.mkdir(exist_ok=True)

    model_files = [
        "heart_disease_model.joblib",
        "pca_model.pkl"
    ]

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backed_up = []
    for model_file in model_files:
        source = model_dir / model_file
        if source.exists():
            backup_name = f"{model_file}_{timestamp}.backup"
            backup_path = backup_dir / backup_name

            try:
                import shutil
                shutil.copy2(source, backup_path)
                backed_up.append(backup_name)
                print(f"✅ Backed up {model_file} to {backup_name}")
            except Exception as e:
                print(f"⚠️  Failed to backup {model_file}: {e}")

    return backed_up


def run_feature_engineering():
    """Run feature engineering if needed"""
    print("🔧 Running feature engineering...")

    try:
        # Import and run build_features
        from src.features.build_features import main as build_features_main
        build_features_main()
        print("✅ Feature engineering completed")
        return True
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_model_training():
    """Run model training"""
    print("🤖 Training new models...")

    try:
        # Import and run train_model
        from src.models.train_model import main as train_model_main
        train_model_main()
        print("✅ Model training completed")
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_new_models():
    """Verify that new models work correctly"""
    print("✅ Verifying new models...")

    try:
        from app.routes import load_model, predict_single_sample, format_prediction_results

        # Load the new model
        model_dict = load_model()
        if model_dict is None:
            print("❌ Failed to load new model")
            return False

        print(f"✅ Model loaded successfully: {type(model_dict['model'])}")

        # Test prediction with sample data
        sample_data = {
            'age': 50,
            'sex': 1,
            'ethnicity': 0,
            'family_history': 1,
            'previous_cardiac_events': 0,
            'diabetes_status': 0,
            'hypertension': 1,
            'systolic_bp': 140,
            'diastolic_bp': 90,
            'resting_heart_rate': 80,
            'height_cm': 175,
            'weight_kg': 80,
            'bmi': 26.1,
            'total_cholesterol': 200,
            'hdl': 45,
            'ldl': 130,
            'fasting_glucose': 95,
            'hba1c': 5.5,
            'smoking_status': 0,
            'alcohol_per_week': 2,
            'physical_activity_hours': 3,
            'diet_quality': 3,
            'stress_level': 2,
            'chest_pain_type': 0,
            'exercise_induced_angina': 0,
            'resting_ecg': 0,
            'max_heart_rate': 160,
            'st_depression': 0.0,
            'st_slope': 1,
            'num_vessels': 0,
            'thalassemia': 1
        }

        prediction, probability = predict_single_sample(model_dict, sample_data)
        result = format_prediction_results(prediction, probability)

        print(f"✅ Test prediction successful:")
        print(f"   Prediction: {prediction}")
        print(f"   Probability: {probability:.1%}")
        print(f"   Risk Level: {result['risk_level']}")

        return True

    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main regeneration process"""
    print("🔄 Model Regeneration Script")
    print("=" * 50)

    # Step 1: Check training data
    data_available, missing_files = check_training_data()
    if not data_available:
        print("\n❌ Cannot regenerate models - missing training data!")
        print("You need to run the data processing pipeline first:")
        for file in missing_files:
            print(f"   Missing: {file}")
        print("\nSuggested steps:")
        print("1. Ensure raw data is in data/raw/")
        print("2. Run: python src/data/preprocess.py")
        print("3. Run: python src/features/build_features.py")
        print("4. Run this script again")
        return False

    # Step 2: Backup existing models
    backed_up = backup_existing_models()

    # Step 3: Check if featured data exists, if not run feature engineering
    featured_data_path = project_root / "data" / "processed" / "featured_heart_data.csv"
    if not featured_data_path.exists():
        print("\n🔧 Featured data not found, running feature engineering...")
        if not run_feature_engineering():
            print("❌ Feature engineering failed - cannot continue")
            return False
    else:
        print("✅ Featured data already exists")

    # Step 4: Train new models
    print("\n🤖 Starting model training...")
    if not run_model_training():
        print("❌ Model training failed")
        return False

    # Step 5: Verify new models
    print("\n✅ Verifying new models...")
    if not verify_new_models():
        print("❌ Model verification failed")
        return False

    print("\n🎉 Model regeneration completed successfully!")
    print("=" * 50)
    print("✅ New models have been generated and verified")
    if backed_up:
        print(f"✅ Old models backed up: {', '.join(backed_up)}")
    print("✅ Your application should now work correctly")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 If you continue to have issues, try:")
        print("1. Check that all required data files exist")
        print("2. Ensure all dependencies are installed")
        print("3. Run each step manually:")
        print("   - python src/data/preprocess.py")
        print("   - python src/features/build_features.py")
        print("   - python src/models/train_model.py")
        sys.exit(1)
    else:
        sys.exit(0)