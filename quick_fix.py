#!/usr/bin/env python3
"""
quick_fix.py - Quick fix for the corrupted model files

This script will:
1. Check if the model files are corrupted
2. If they are, it will try to regenerate them
3. If that fails, create a simple dummy model for testing
"""

import os
import sys
from pathlib import Path
import joblib
import pickle

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def check_model_corruption():
    """Check if model files are corrupted"""
    print("🔍 Checking model file integrity...")

    model_path = project_root / "data" / "models" / "heart_disease_model.joblib"
    pca_path = project_root / "data" / "models" / "pca_model.pkl"

    issues = []

    # Check main model
    if not model_path.exists():
        issues.append("Model file missing")
    else:
        try:
            model = joblib.load(model_path)
            print("✅ Main model loads successfully")
        except Exception as e:
            issues.append(f"Model file corrupted: {e}")

    # Check PCA model
    if not pca_path.exists():
        print("⚠️  PCA model missing (optional)")
    else:
        try:
            with open(pca_path, 'rb') as f:
                pca = pickle.load(f)
            print("✅ PCA model loads successfully")
        except Exception as e:
            issues.append(f"PCA model corrupted: {e}")

    return issues


def try_regenerate_models():
    """Try to regenerate models using existing training data"""
    print("🔄 Attempting to regenerate models...")

    # Check if training data exists
    featured_data_path = project_root / "data" / "processed" / "featured_heart_data.csv"

    if not featured_data_path.exists():
        print("❌ No training data found - cannot regenerate models")
        return False

    try:
        # Try to run the training script
        from src.models.train_model import main as train_main
        train_main()
        print("✅ Models regenerated successfully")
        return True
    except Exception as e:
        print(f"❌ Model regeneration failed: {e}")
        return False


def create_dummy_model():
    """Create a simple dummy model for testing purposes"""
    print("🔧 Creating dummy model for testing...")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.decomposition import PCA
        import numpy as np

        # Create a simple dummy model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Create some dummy training data to fit the model
        X_dummy = np.random.rand(100, 32)  # 32 features
        y_dummy = np.random.randint(0, 2, 100)

        model.fit(X_dummy, y_dummy)

        # Save the dummy model
        model_dir = project_root / "data" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "heart_disease_model.joblib"
        joblib.dump(model, model_path)

        # Create a simple PCA model too
        pca = PCA(n_components=5)
        pca.fit(X_dummy)

        pca_path = model_dir / "pca_model.pkl"
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)

        print("✅ Dummy models created successfully")
        print("⚠️  These are dummy models for testing only!")
        print("   You should regenerate proper models when possible")

        return True

    except Exception as e:
        print(f"❌ Failed to create dummy models: {e}")
        return False


def test_model_loading():
    """Test if models can now be loaded"""
    print("🧪 Testing model loading...")

    try:
        from app.routes import load_model
        model_dict = load_model()

        if model_dict and 'model' in model_dict:
            print("✅ Models load successfully")
            return True
        else:
            print("❌ Model loading still fails")
            return False

    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def main():
    """Main fix process"""
    print("🔧 Quick Fix for Heart Disease Prediction Models")
    print("=" * 55)

    # Step 1: Check for corruption
    issues = check_model_corruption()

    if not issues:
        print("✅ All models are working correctly!")
        return True

    print(f"❌ Found {len(issues)} issues:")
    for issue in issues:
        print(f"   - {issue}")

    # Step 2: Try to regenerate models
    print("\n🔄 Attempting to fix issues...")

    if try_regenerate_models():
        print("✅ Models successfully regenerated!")
        return test_model_loading()

    # Step 3: Create dummy models as last resort
    print("\n🔧 Regeneration failed, creating dummy models for testing...")

    if create_dummy_model():
        if test_model_loading():
            print("\n✅ Quick fix completed!")
            print("Your app should now work for testing purposes.")
            print("\n⚠️  IMPORTANT: The current models are dummy models.")
            print("For production use, you need to:")
            print("1. Ensure you have proper training data")
            print("2. Run: python regenerate_models.py")
            return True

    print("\n❌ Quick fix failed!")
    print("Manual intervention required:")
    print("1. Check that all dependencies are installed")
    print("2. Ensure training data exists in data/processed/")
    print("3. Try running: python src/models/train_model.py")

    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)