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

    # Fix: Add a placeholder option and adjust validation
    sex = SelectField('Sex', choices=[
        ('', 'Select Sex'),  # Placeholder
        ('1', 'Male'),
        ('0', 'Female')
    ], validators=[DataRequired(message='Please select a sex')])

    ethnicity = SelectField('Ethnicity', choices=[
        ('', 'Select Ethnicity'),  # Placeholder
        ('0', 'White'),
        ('1', 'Black'),
        ('2', 'Hispanic'),
        ('3', 'Asian'),
        ('4', 'Other')
    ], validators=[DataRequired(message='Please select an ethnicity')])

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
        ('', 'Select Smoking Status'),  # Placeholder
        ('0', 'Never'),
        ('1', 'Former'),
        ('2', 'Current')
    ], validators=[DataRequired(message='Please select smoking status')])

    alcohol_per_week = IntegerField('Alcohol Units per Week', validators=[
        DataRequired(),
        NumberRange(min=0, max=50, message='Please enter units per week (0-50)')
    ])

    physical_activity_hours = FloatField('Physical Activity (hours/week)', validators=[
        DataRequired(),
        NumberRange(min=0, max=50, message='Please enter hours per week (0-50)')
    ])

    diet_quality = SelectField('Diet Quality', choices=[
        ('', 'Select Diet Quality'),  # Placeholder
        ('1', 'Poor'),
        ('2', 'Fair'),
        ('3', 'Good'),
        ('4', 'Excellent')
    ], validators=[DataRequired(message='Please select diet quality')])

    stress_level = SelectField('Stress Level', choices=[
        ('', 'Select Stress Level'),  # Placeholder
        ('1', 'Very Low'),
        ('2', 'Low'),
        ('3', 'Moderate'),
        ('4', 'High'),
        ('5', 'Very High')
    ], validators=[DataRequired(message='Please select stress level')])

    # Clinical Tests
    chest_pain_type = SelectField('Chest Pain Type', choices=[
        ('', 'Select Chest Pain Type'),  # Placeholder
        ('0', 'Typical Angina'),
        ('1', 'Atypical Angina'),
        ('2', 'Non-anginal Pain'),
        ('3', 'Asymptomatic')
    ], validators=[DataRequired(message='Please select chest pain type')])

    exercise_induced_angina = BooleanField('Exercise Induced Angina')

    resting_ecg = SelectField('Resting ECG Results', choices=[
        ('', 'Select ECG Results'),  # Placeholder
        ('0', 'Normal'),
        ('1', 'ST-T Wave Abnormality'),
        ('2', 'Left Ventricular Hypertrophy')
    ], validators=[DataRequired(message='Please select ECG results')])

    max_heart_rate = IntegerField('Maximum Heart Rate Achieved', validators=[
        DataRequired(),
        NumberRange(min=60, max=220, message='Please enter a valid heart rate between 60 and 220 bpm')
    ])

    st_depression = FloatField('ST Depression Induced by Exercise', validators=[
        DataRequired(),
        NumberRange(min=0, max=10, message='Please enter a valid ST depression value between 0 and 10')
    ])

    st_slope = SelectField('Slope of Peak Exercise ST Segment', choices=[
        ('', 'Select ST Slope'),  # Placeholder
        ('0', 'Upsloping'),
        ('1', 'Flat'),
        ('2', 'Downsloping')
    ], validators=[DataRequired(message='Please select ST slope')])

    num_vessels = SelectField('Number of Major Vessels Colored by Fluoroscopy', choices=[
        ('', 'Select Number'),  # Placeholder
        ('0', '0'),
        ('1', '1'),
        ('2', '2'),
        ('3', '3'),
        ('4', '4')
    ], validators=[DataRequired(message='Please select number of vessels')])

    thalassemia = SelectField('Thalassemia', choices=[
        ('', 'Select Thalassemia Type'),  # Placeholder
        ('1', 'Normal'),
        ('2', 'Fixed Defect'),
        ('3', 'Reversible Defect')
    ], validators=[DataRequired(message='Please select thalassemia type')])

    submit = SubmitField('Predict')