import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import random

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app import create_app
from app.models import db, Prediction, PredictionSummary, ModelVersion, User
from sqlalchemy import inspect

def create_tables():
    """Create all database tables"""
    print("Creating database tables...")

    try:
        db.create_all()
        print("✅ Database tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        return False


def seed_users():
    """Seed initial user data"""
    print("Seeding users...")

    try:
        # Check if users already exist
        if User.query.first():
            print("  Users already exist, skipping...")
            return True

        # Create initial admin user
        admin = User(username="admin", email="admin@heartdisease.com", role="admin")
        admin.set_password("admin123")  # Change this in production!
        db.session.add(admin)

        # Create initial analyst user
        analyst = User(username="analyst", email="analyst@heartdisease.com", role="analyst")
        analyst.set_password("analyst123")  # Change this in production!
        db.session.add(analyst)

        # Create a sample clinician user for testing
        clinician = User(username="clinician1", email="clinician@heartdisease.com", role="clinician")
        clinician.set_password("clinician123")  # Change this in production!
        db.session.add(clinician)

        db.session.commit()
        print("✅ Users seeded successfully!")
        print("   Default credentials:")
        print("   Admin: admin / admin123")
        print("   Analyst: analyst / analyst123")
        print("   Clinician: clinician1 / clinician123")
        print("   ⚠️  Please change these passwords in production!")
        return True

    except Exception as e:
        print(f"❌ Error seeding users: {e}")
        db.session.rollback()
        return False


def seed_model_versions():
    """Seed initial model version data"""
    print("Seeding model versions...")

    try:
        # Check if model versions already exist
        if ModelVersion.query.first():
            print("  Model versions already exist, skipping...")
            return True

        # Create initial model version
        model_v1 = ModelVersion(
            version="1.0",
            description="Initial XGBoost model with comprehensive feature engineering",
            accuracy=0.92,
            precision=0.89,
            recall=0.94,
            f1_score=0.91,
            roc_auc=0.95,
            model_path="data/models/heart_disease_model.joblib",
            pca_model_path="data/models/pca_model.pkl",
            training_date=datetime.utcnow() - timedelta(days=30),
            is_active=True
        )

        db.session.add(model_v1)
        db.session.commit()

        print("✅ Model versions seeded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error seeding model versions: {e}")
        db.session.rollback()
        return False


def seed_sample_predictions(num_samples=50):
    """Seed database with sample prediction data for testing"""
    print(f"Seeding {num_samples} sample predictions...")

    try:
        # Check if predictions already exist
        if Prediction.query.first():
            print("  Predictions already exist, skipping...")
            return True

        # Generate sample predictions
        for i in range(num_samples):
            # Random patient data
            age = random.randint(25, 85)
            sex = random.choice([0, 1])

            # Generate correlated health data
            has_diabetes = random.choice([True, False])
            has_hypertension = random.choice([True, False])
            is_smoker = random.choice([0, 1, 2])  # Never, Former, Current

            # Generate vital signs with some correlation
            base_systolic = 120 + (20 if has_hypertension else 0) + random.randint(-15, 15)
            base_diastolic = 80 + (10 if has_hypertension else 0) + random.randint(-10, 10)

            # Generate lab values
            cholesterol = random.randint(150, 300)
            hdl = random.randint(30, 80)
            ldl = cholesterol - hdl - random.randint(20, 40)

            # Calculate risk-based probability
            risk_factors = 0
            risk_factors += (age - 40) / 10 if age > 40 else 0
            risk_factors += 2 if sex == 1 else 0  # Male
            risk_factors += 3 if has_diabetes else 0
            risk_factors += 2 if has_hypertension else 0
            risk_factors += 2 if is_smoker == 2 else (1 if is_smoker == 1 else 0)
            risk_factors += (cholesterol - 200) / 50 if cholesterol > 200 else 0

            # Convert to probability (sigmoid-like function)
            probability = min(0.95, max(0.05, 1 / (1 + 2.718 ** (-(risk_factors - 5)))))
            prediction_result = 1 if probability > 0.5 else 0

            # Determine risk level
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

            # Create prediction record
            prediction = Prediction(
                created_at=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                age=age,
                sex=sex,
                ethnicity=random.choice([0, 1, 2, 3, 4]),
                family_history=random.choice([True, False]),
                previous_cardiac_events=random.choice([True, False]),
                diabetes_status=has_diabetes,
                hypertension=has_hypertension,
                systolic_bp=base_systolic,
                diastolic_bp=base_diastolic,
                resting_heart_rate=random.randint(60, 100),
                height_cm=random.randint(150, 190),
                weight_kg=random.randint(50, 120),
                bmi=random.uniform(18.5, 35.0),
                total_cholesterol=cholesterol,
                hdl=hdl,
                ldl=ldl,
                fasting_glucose=random.randint(80, 180),
                hba1c=random.uniform(4.5, 9.0),
                smoking_status=is_smoker,
                alcohol_per_week=random.randint(0, 20),
                physical_activity_hours=random.uniform(0, 15),
                diet_quality=random.choice([1, 2, 3, 4]),
                stress_level=random.choice([1, 2, 3, 4, 5]),
                chest_pain_type=random.choice([0, 1, 2, 3]),
                exercise_induced_angina=random.choice([True, False]),
                resting_ecg=random.choice([0, 1, 2]),
                max_heart_rate=random.randint(100, 200),
                st_depression=random.uniform(0, 4),
                st_slope=random.choice([0, 1, 2]),
                num_vessels=random.choice([0, 1, 2, 3, 4]),
                thalassemia=random.choice([1, 2, 3]),
                prediction_result=prediction_result,
                probability=probability,
                risk_level=risk_level,
                interpretation=f"The model predicts {'heart disease' if prediction_result == 1 else 'no heart disease'} with a probability of {probability:.2%}. This represents a {risk_level} risk level.",
                model_version="1.0",
                session_id=f"test_session_{i}",
                ip_address="127.0.0.1"
            )

            db.session.add(prediction)

        db.session.commit()
        print(f"✅ {num_samples} sample predictions seeded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error seeding sample predictions: {e}")
        db.session.rollback()
        return False


def create_summary_data():
    """Create summary data for existing predictions"""
    print("Creating prediction summaries...")

    try:
        # Get all prediction dates
        prediction_dates = db.session.query(
            db.func.date(Prediction.created_at).label('date')
        ).distinct().all()

        for date_row in prediction_dates:
            # FIXED: Convert string date to Python date object
            date_str = date_row[0]
            if isinstance(date_str, str):
                # Parse string date into date object
                pred_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            else:
                # Already a date object
                pred_date = date_str

            # Check if summary already exists
            if PredictionSummary.query.filter_by(date=pred_date).first():
                continue

            # Get predictions for this date
            daily_predictions = Prediction.query.filter(
                db.func.date(Prediction.created_at) == date_str  # Use string for comparison
            ).all()

            if not daily_predictions:
                continue

            # Calculate summary statistics
            total = len(daily_predictions)
            positive = sum(1 for p in daily_predictions if p.prediction_result == 1)
            negative = total - positive

            avg_probability = sum(p.probability for p in daily_predictions) / total
            avg_age = sum(p.age for p in daily_predictions) / total

            # Count risk levels
            risk_counts = {'very_low': 0, 'low': 0, 'moderate': 0, 'high': 0, 'very_high': 0}
            for prediction in daily_predictions:
                risk_key = prediction.risk_level.lower().replace(' ', '_')
                if risk_key in risk_counts:
                    risk_counts[risk_key] += 1

            # Create summary with proper date object
            summary = PredictionSummary(
                date=pred_date,  # Now using the converted date object
                total_predictions=total,
                positive_predictions=positive,
                negative_predictions=negative,
                avg_probability=avg_probability,
                avg_age=avg_age,
                risk_very_low=risk_counts['very_low'],
                risk_low=risk_counts['low'],
                risk_moderate=risk_counts['moderate'],
                risk_high=risk_counts['high'],
                risk_very_high=risk_counts['very_high']
            )

            db.session.add(summary)

        db.session.commit()
        print("✅ Prediction summaries created successfully!")
        return True

    except Exception as e:
        print(f"❌ Error creating prediction summaries: {e}")
        db.session.rollback()
        return False


def verify_database():
    """Verify database structure and data"""
    print("Verifying database structure...")

    try:
        inspector = inspect(db.engine)
        expected_tables = ['users', 'predictions', 'prediction_summaries', 'model_versions']

        for table in expected_tables:
            if inspector.has_table(table):
                print(f"✅ Table '{table}' exists")
            else:
                print(f"❌ Table '{table}' missing")
                return False

        # Check data counts
        user_count = User.query.count()
        prediction_count = Prediction.query.count()
        summary_count = PredictionSummary.query.count()
        model_count = ModelVersion.query.count()

        print(f"✅ Database contains:")
        print(f"  - {user_count} users")
        print(f"  - {prediction_count} predictions")
        print(f"  - {summary_count} daily summaries")
        print(f"  - {model_count} model versions")

        return True

    except Exception as e:
        print(f"❌ Error verifying database: {e}")
        return False


def main():
    """Main initialization function"""
    print("Heart Disease Prediction System - Database Initialization")
    print("=" * 60)

    # Create Flask app context
    app = create_app('development')

    with app.app_context():
        success = True

        # Create tables
        if not create_tables():
            success = False

        # Seed users first
        if success and not seed_users():
            success = False

        # Seed model versions
        if success and not seed_model_versions():
            success = False

        # Ask user if they want to seed sample data
        if success:
            seed_data = input("\nDo you want to seed sample prediction data? (y/n): ").lower().strip()
            if seed_data == 'y':
                if not seed_sample_predictions(50):
                    success = False
                elif not create_summary_data():
                    success = False

        # Verify database
        if success:
            success = verify_database()

        if success:
            print("\n" + "=" * 60)
            print("✅ Database initialization completed successfully!")
            print("\nDefault User Accounts Created:")
            print("  Admin:     admin / admin123")
            print("  Analyst:   analyst / analyst123") 
            print("  Clinician: clinician1 / clinician123")
            print("\n⚠️  IMPORTANT: Change these default passwords in production!")
            print("\nYou can now run the application with:")
            print("  python run.py")
        else:
            print("\n" + "=" * 60)
            print("❌ Database initialization failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()