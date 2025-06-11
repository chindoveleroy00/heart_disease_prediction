from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from app.forms import PredictionForm

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import prediction functions
try:
    from src.models.predict_model import predict_single_sample, format_prediction_results, load_model
except ImportError:
    # Fallback if the module structure is different
    print("Warning: Could not import predict_model functions. Using fallback prediction.")

main = Blueprint('main', __name__)

# Global model variable to avoid reloading
_model = None

def get_model():
    """Load model once and cache it"""
    global _model
    if _model is None:
        try:
            _model = load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            _model = None
    return _model

@main.route('/')
def index():
    """Home page route"""
    return render_template('index.html', title='Heart Disease Prediction')

@main.route('/about')
def about():
    """About page route"""
    return render_template('about.html', title='About')

@main.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction form route using the proper feature engineering pipeline"""
    form = PredictionForm()
    
    if form.validate_on_submit():
        try:
            # Create input data dictionary with all required features
            input_data = {
                'age': form.age.data,
                'sex': form.sex.data,
                'ethnicity': form.ethnicity.data,
                'family_history': 1 if form.family_history.data else 0,
                'previous_cardiac_events': 1 if form.previous_cardiac_events.data else 0,
                'diabetes_status': 1 if form.diabetes_status.data else 0,
                'hypertension': 1 if form.hypertension.data else 0,
                'systolic_bp': form.systolic_bp.data,
                'diastolic_bp': form.diastolic_bp.data,
                'resting_heart_rate': form.resting_heart_rate.data,
                'height_cm': form.height_cm.data,
                'weight_kg': form.weight_kg.data,
                'bmi': form.weight_kg.data / ((form.height_cm.data / 100) ** 2),  # Calculate BMI
                'total_cholesterol': form.total_cholesterol.data,
                'hdl': form.hdl.data,
                'ldl': form.ldl.data,
                'fasting_glucose': form.fasting_glucose.data,
                'hba1c': form.hba1c.data,
                'smoking_status': form.smoking_status.data,
                'alcohol_per_week': form.alcohol_per_week.data,
                'physical_activity_hours': form.physical_activity_hours.data,
                'diet_quality': form.diet_quality.data,
                'stress_level': form.stress_level.data,
                'chest_pain_type': form.chest_pain_type.data,
                'exercise_induced_angina': 1 if form.exercise_induced_angina.data else 0,
                'resting_ecg': form.resting_ecg.data,
                'max_heart_rate': form.max_heart_rate.data,
                'st_depression': form.st_depression.data,
                'st_slope': form.st_slope.data,
                'num_vessels': form.num_vessels.data,
                'thalassemia': form.thalassemia.data
            }
            
            # Load model
            model = get_model()
            if model is None:
                flash('Model is currently unavailable. Please try again later.', 'error')
                return render_template('predict.html', title='Make Prediction', form=form)
            
            # Make prediction using the proper pipeline
            prediction, probability = predict_single_sample(model, input_data)
            result = format_prediction_results(prediction, probability)
            
            # Return results
            return render_template(
                'results.html',
                title='Prediction Results',
                prediction=result['prediction'],
                probability=result['probability'],
                risk_level=result['risk_level'],
                interpretation=result['interpretation'],
                data=input_data
            )
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            flash(f'An error occurred during prediction: {str(e)}', 'error')
            return render_template('predict.html', title='Make Prediction', form=form)
    
    return render_template('predict.html', title='Make Prediction', form=form)

@main.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'age', 'sex', 'systolic_bp', 'diastolic_bp', 'resting_heart_rate',
            'height_cm', 'weight_kg', 'total_cholesterol', 'hdl', 'ldl',
            'fasting_glucose', 'hba1c'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Calculate BMI if not provided
        if 'bmi' not in data:
            data['bmi'] = data['weight_kg'] / ((data['height_cm'] / 100) ** 2)
        
        # Set defaults for optional fields
        defaults = {
            'ethnicity': 0,
            'family_history': 0,
            'previous_cardiac_events': 0,
            'diabetes_status': 0,
            'hypertension': 0,
            'smoking_status': 0,
            'alcohol_per_week': 0,
            'physical_activity_hours': 2,
            'diet_quality': 2,
            'stress_level': 3,
            'chest_pain_type': 0,
            'exercise_induced_angina': 0,
            'resting_ecg': 0,
            'max_heart_rate': 150,
            'st_depression': 0.0,
            'st_slope': 1,
            'num_vessels': 0,
            'thalassemia': 1
        }
        
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        
        # Load model and make prediction
        model = get_model()
        if model is None:
            return jsonify({'error': 'Model is currently unavailable'}), 503
        
        prediction, probability = predict_single_sample(model, data)
        result = format_prediction_results(prediction, probability)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@main.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        status = 'healthy' if model is not None else 'unhealthy'
        return jsonify({'status': status, 'model_loaded': model is not None})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500