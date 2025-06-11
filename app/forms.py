from flask_wtf import FlaskForm
from wtforms import (
    StringField, IntegerField, FloatField, SelectField, 
    BooleanField, SubmitField
)
from wtforms.validators import DataRequired, NumberRange

class PredictionForm(FlaskForm):
    # Demographics
    age = IntegerField('Age', validators=[
        DataRequired(), 
        NumberRange(min=18, max=120, message='Please enter a valid age between 18 and 120')
    ])
    
    sex = SelectField('Sex', choices=[
        (1, 'Male'), 
        (0, 'Female')
    ], coerce=int, validators=[DataRequired()])
    
    ethnicity = SelectField('Ethnicity', choices=[
        (0, 'White'),
        (1, 'Black'),
        (2, 'Hispanic'),
        (3, 'Asian'),
        (4, 'Other')
    ], coerce=int, validators=[DataRequired()])
    
    # Medical History
    family_history = BooleanField('Family History of Heart Disease')
    previous_cardiac_events = BooleanField('Previous Cardiac Events')
    diabetes_status = BooleanField('Diabetes')
    hypertension = BooleanField('Hypertension')
    
    # Vital Signs
    systolic_bp = IntegerField('Systolic Blood Pressure (mmHg)', validators=[
        DataRequired(),
        NumberRange(min=80, max=220, message='Please enter a valid systolic BP between 80 and 220 mmHg')
    ])
    
    diastolic_bp = IntegerField('Diastolic Blood Pressure (mmHg)', validators=[
        DataRequired(),
        NumberRange(min=40, max=120, message='Please enter a valid diastolic BP between 40 and 120 mmHg')
    ])
    
    resting_heart_rate = IntegerField('Resting Heart Rate (bpm)', validators=[
        DataRequired(),
        NumberRange(min=40, max=120, message='Please enter a valid heart rate between 40 and 120 bpm')
    ])
    
    # Physical Measurements
    height_cm = IntegerField('Height (cm)', validators=[
        DataRequired(),
        NumberRange(min=120, max=220, message='Please enter a valid height between 120 and 220 cm')
    ])
    
    weight_kg = FloatField('Weight (kg)', validators=[
        DataRequired(),
        NumberRange(min=30, max=200, message='Please enter a valid weight between 30 and 200 kg')
    ])
    
    # Lab Values
    total_cholesterol = IntegerField('Total Cholesterol (mg/dl)', validators=[
        DataRequired(),
        NumberRange(min=100, max=600, message='Please enter a valid cholesterol level between 100 and 600 mg/dl')
    ])
    
    hdl = IntegerField('HDL Cholesterol (mg/dl)', validators=[
        DataRequired(),
        NumberRange(min=20, max=100, message='Please enter a valid HDL level between 20 and 100 mg/dl')
    ])
    
    ldl = IntegerField('LDL Cholesterol (mg/dl)', validators=[
        DataRequired(),
        NumberRange(min=50, max=400, message='Please enter a valid LDL level between 50 and 400 mg/dl')
    ])
    
    fasting_glucose = IntegerField('Fasting Glucose (mg/dl)', validators=[
        DataRequired(),
        NumberRange(min=50, max=400, message='Please enter a valid glucose level between 50 and 400 mg/dl')
    ])
    
    hba1c = FloatField('HbA1c (%)', validators=[
        DataRequired(),
        NumberRange(min=4.0, max=15.0, message='Please enter a valid HbA1c between 4.0 and 15.0%')
    ])
    
    # Lifestyle Factors
    smoking_status = SelectField('Smoking Status', choices=[
        (0, 'Never'),
        (1, 'Former'),
        (2, 'Current')
    ], coerce=int, validators=[DataRequired()])
    
    alcohol_per_week = IntegerField('Alcohol Units per Week', validators=[
        DataRequired(),
        NumberRange(min=0, max=50, message='Please enter units per week (0-50)')
    ])
    
    physical_activity_hours = FloatField('Physical Activity (hours/week)', validators=[
        DataRequired(),
        NumberRange(min=0, max=50, message='Please enter hours per week (0-50)')
    ])
    
    diet_quality = SelectField('Diet Quality', choices=[
        (1, 'Poor'),
        (2, 'Fair'),
        (3, 'Good'),
        (4, 'Excellent')
    ], coerce=int, validators=[DataRequired()])
    
    stress_level = SelectField('Stress Level', choices=[
        (1, 'Very Low'),
        (2, 'Low'),
        (3, 'Moderate'),
        (4, 'High'),
        (5, 'Very High')
    ], coerce=int, validators=[DataRequired()])
    
    # Clinical Tests
    chest_pain_type = SelectField('Chest Pain Type', choices=[
        (0, 'Typical Angina'),
        (1, 'Atypical Angina'),
        (2, 'Non-anginal Pain'),
        (3, 'Asymptomatic')
    ], coerce=int, validators=[DataRequired()])
    
    exercise_induced_angina = BooleanField('Exercise Induced Angina')
    
    resting_ecg = SelectField('Resting ECG Results', choices=[
        (0, 'Normal'),
        (1, 'ST-T Wave Abnormality'),
        (2, 'Left Ventricular Hypertrophy')
    ], coerce=int, validators=[DataRequired()])
    
    max_heart_rate = IntegerField('Maximum Heart Rate Achieved', validators=[
        DataRequired(),
        NumberRange(min=60, max=220, message='Please enter a valid heart rate between 60 and 220 bpm')
    ])
    
    st_depression = FloatField('ST Depression Induced by Exercise', validators=[
        DataRequired(),
        NumberRange(min=0, max=10, message='Please enter a valid ST depression value between 0 and 10')
    ])
    
    st_slope = SelectField('Slope of Peak Exercise ST Segment', choices=[
        (0, 'Upsloping'),
        (1, 'Flat'),
        (2, 'Downsloping')
    ], coerce=int, validators=[DataRequired()])
    
    num_vessels = SelectField('Number of Major Vessels Colored by Fluoroscopy', choices=[
        (0, '0'),
        (1, '1'),
        (2, '2'),
        (3, '3'),
        (4, '4')
    ], coerce=int, validators=[DataRequired()])
    
    thalassemia = SelectField('Thalassemia', choices=[
        (1, 'Normal'),
        (2, 'Fixed Defect'),
        (3, 'Reversible Defect')
    ], coerce=int, validators=[DataRequired()])
    
    submit = SubmitField('Predict')