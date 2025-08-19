from app import db
from datetime import datetime
import json

class Prediction(db.Model):
    """Model to store heart disease predictions with all parameters and results"""

    __tablename__ = 'predictions'

    # Primary key
    id = db.Column(db.Integer, primary_key=True)

    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Patient identifier (optional - for linking predictions to patients)
    patient_id = db.Column(db.String(100), nullable=True, index=True)
    session_id = db.Column(db.String(100), nullable=True, index=True)

    # Demographics
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)  # 0=Female, 1=Male
    ethnicity = db.Column(db.Integer, nullable=False)  # 0=White, 1=Black, etc.

    # Medical History
    family_history = db.Column(db.Boolean, nullable=False, default=False)
    previous_cardiac_events = db.Column(db.Boolean, nullable=False, default=False)
    diabetes_status = db.Column(db.Boolean, nullable=False, default=False)
    hypertension = db.Column(db.Boolean, nullable=False, default=False)

    # Vital Signs
    systolic_bp = db.Column(db.Integer, nullable=False)
    diastolic_bp = db.Column(db.Integer, nullable=False)
    resting_heart_rate = db.Column(db.Integer, nullable=False)

    # Physical Measurements
    height_cm = db.Column(db.Integer, nullable=False)
    weight_kg = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)

    # Laboratory Values
    total_cholesterol = db.Column(db.Integer, nullable=False)
    hdl = db.Column(db.Integer, nullable=False)
    ldl = db.Column(db.Integer, nullable=False)
    fasting_glucose = db.Column(db.Integer, nullable=False)
    hba1c = db.Column(db.Float, nullable=False)

    # Lifestyle Factors
    smoking_status = db.Column(db.Integer, nullable=False)  # 0=Never, 1=Former, 2=Current
    alcohol_per_week = db.Column(db.Integer, nullable=False, default=0)
    physical_activity_hours = db.Column(db.Float, nullable=False, default=0)
    diet_quality = db.Column(db.Integer, nullable=False)  # 1-4 scale
    stress_level = db.Column(db.Integer, nullable=False)  # 1-5 scale

    # Clinical Tests
    chest_pain_type = db.Column(db.Integer, nullable=False)  # 0-3
    exercise_induced_angina = db.Column(db.Boolean, nullable=False, default=False)
    resting_ecg = db.Column(db.Integer, nullable=False)  # 0-2
    max_heart_rate = db.Column(db.Integer, nullable=False)
    st_depression = db.Column(db.Float, nullable=False)
    st_slope = db.Column(db.Integer, nullable=False)  # 0-2
    num_vessels = db.Column(db.Integer, nullable=False)  # 0-4
    thalassemia = db.Column(db.Integer, nullable=False)  # 1-3

    # Prediction Results
    prediction_result = db.Column(db.Integer, nullable=False)  # 0=No Disease, 1=Disease
    probability = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    risk_level = db.Column(db.String(20), nullable=False)  # Very Low, Low, Moderate, High, Very High
    interpretation = db.Column(db.Text, nullable=False)

    # Model Information
    model_version = db.Column(db.String(20), nullable=True)
    model_features_used = db.Column(db.Text, nullable=True)  # JSON string of features

    # Additional metadata
    ip_address = db.Column(db.String(45), nullable=True)  # For audit trail
    user_agent = db.Column(db.String(500), nullable=True)

    def __repr__(self):
        return f'<Prediction {self.id}: {self.risk_level} ({self.probability:.2%})>'

    def to_dict(self):
        """Convert prediction to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'patient_id': self.patient_id,
            'demographics': {
                'age': self.age,
                'sex': 'Male' if self.sex == 1 else 'Female',
                'ethnicity': self.get_ethnicity_label()
            },
            'medical_history': {
                'family_history': self.family_history,
                'previous_cardiac_events': self.previous_cardiac_events,
                'diabetes_status': self.diabetes_status,
                'hypertension': self.hypertension
            },
            'vital_signs': {
                'systolic_bp': self.systolic_bp,
                'diastolic_bp': self.diastolic_bp,
                'resting_heart_rate': self.resting_heart_rate
            },
            'physical_measurements': {
                'height_cm': self.height_cm,
                'weight_kg': self.weight_kg,
                'bmi': round(self.bmi, 1)
            },
            'lab_values': {
                'total_cholesterol': self.total_cholesterol,
                'hdl': self.hdl,
                'ldl': self.ldl,
                'fasting_glucose': self.fasting_glucose,
                'hba1c': self.hba1c
            },
            'lifestyle': {
                'smoking_status': self.get_smoking_label(),
                'alcohol_per_week': self.alcohol_per_week,
                'physical_activity_hours': self.physical_activity_hours,
                'diet_quality': self.get_diet_quality_label(),
                'stress_level': self.get_stress_level_label()
            },
            'clinical_tests': {
                'chest_pain_type': self.get_chest_pain_label(),
                'exercise_induced_angina': self.exercise_induced_angina,
                'resting_ecg': self.get_ecg_label(),
                'max_heart_rate': self.max_heart_rate,
                'st_depression': self.st_depression,
                'st_slope': self.get_st_slope_label(),
                'num_vessels': self.num_vessels,
                'thalassemia': self.get_thalassemia_label()
            },
            'results': {
                'prediction': 'Heart Disease' if self.prediction_result == 1 else 'No Heart Disease',
                'probability': round(self.probability * 100, 2),
                'risk_level': self.risk_level,
                'interpretation': self.interpretation
            },
            'model_info': {
                'version': self.model_version,
                'features_used': json.loads(self.model_features_used) if self.model_features_used else None
            }
        }

    # Helper methods for human-readable labels
    def get_ethnicity_label(self):
        ethnicity_map = {0: 'White', 1: 'Black', 2: 'Hispanic', 3: 'Asian', 4: 'Other'}
        return ethnicity_map.get(self.ethnicity, 'Unknown')

    def get_smoking_label(self):
        smoking_map = {0: 'Never', 1: 'Former', 2: 'Current'}
        return smoking_map.get(self.smoking_status, 'Unknown')

    def get_diet_quality_label(self):
        quality_map = {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
        return quality_map.get(self.diet_quality, 'Unknown')

    def get_stress_level_label(self):
        stress_map = {1: 'Very Low', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'Very High'}
        return stress_map.get(self.stress_level, 'Unknown')

    def get_chest_pain_label(self):
        pain_map = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
        return pain_map.get(self.chest_pain_type, 'Unknown')

    def get_ecg_label(self):
        ecg_map = {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}
        return ecg_map.get(self.resting_ecg, 'Unknown')

    def get_st_slope_label(self):
        slope_map = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
        return slope_map.get(self.st_slope, 'Unknown')

    def get_thalassemia_label(self):
        thal_map = {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}
        return thal_map.get(self.thalassemia, 'Unknown')


class PredictionSummary(db.Model):
    """Model to store daily prediction summaries for analytics"""

    __tablename__ = 'prediction_summaries'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, unique=True, index=True)

    total_predictions = db.Column(db.Integer, default=0)
    positive_predictions = db.Column(db.Integer, default=0)
    negative_predictions = db.Column(db.Integer, default=0)

    avg_probability = db.Column(db.Float, default=0.0)
    avg_age = db.Column(db.Float, default=0.0)

    risk_very_low = db.Column(db.Integer, default=0)
    risk_low = db.Column(db.Integer, default=0)
    risk_moderate = db.Column(db.Integer, default=0)
    risk_high = db.Column(db.Integer, default=0)
    risk_very_high = db.Column(db.Integer, default=0)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<PredictionSummary {self.date}: {self.total_predictions} predictions>'


class ModelVersion(db.Model):
    """Model to track different model versions and their performance"""

    __tablename__ = 'model_versions'

    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(20), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=True)

    # Model performance metrics
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    roc_auc = db.Column(db.Float, nullable=True)

    # Model file paths
    model_path = db.Column(db.String(500), nullable=True)
    pca_model_path = db.Column(db.String(500), nullable=True)

    # Metadata
    training_date = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<ModelVersion {self.version}: {"Active" if self.is_active else "Inactive"}>'