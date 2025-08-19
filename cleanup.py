# cleanup_and_regenerate.py
import os
import shutil
from pathlib import Path


def cleanup_models():
    model_files = [
        "data/models/heart_disease_model.joblib",
        "data/models/pca_model.pkl"
    ]

    for model_file in model_files:
        path = Path(model_file)
        if path.exists():
            try:
                path.unlink()
                print(f"Deleted {model_file}")
            except Exception as e:
                print(f"Error deleting {model_file}: {e}")


def main():
    print("🧹 Cleaning up old models...")
    cleanup_models()

    print("\n🔄 Regenerating models...")
    os.system("python src/data/preprocess.py")
    os.system("python src/features/build_features.py")
    os.system("python src/models/train_model.py")

    print("\n✅ Cleanup and regeneration complete!")


if __name__ == "__main__":
    main()