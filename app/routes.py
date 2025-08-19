from flask import Blueprint, render_template, request, flash, jsonify, session, redirect, url_for, current_app
import sys
import json
from pathlib import Path
from datetime import datetime, date
from sqlalchemy import func, desc
from app.forms import PredictionForm
from app.models import Prediction, PredictionSummary, ModelVersion
from app import db

# Fallback prediction functions (since import is failing)
import joblib
import pandas as pd
import numpy as np
import pickle

# Define Blueprint at the top of the file
bp = Blueprint('main', __name__)

# Global model variable to avoid reloading
_model = None


def load_model():
    """Load the trained model and PCA transformer"""
    try:
        # Define paths
        base_path = Path(__file__).parent.parent
        model_path = base_path / "data" / "models" / "heart_disease_model.joblib"
        pca_path = base_path / "data" / "models" / "pca_model.pkl"

        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)

        pca_model = None
        if pca_path.exists():
            print(f"Loading PCA model from {pca_path}...")
            pca_model = joblib.load(pca_path)  # Use joblib for both for consistency

        return {'model': model, 'pca': pca_model}
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_input_data(input_data):
    """Preprocess input data to match model expectations including all engineered features."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure all values are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill any NaN values with defaults (important for new features too)
        df = df.fillna(0) # Consider more sophisticated imputation if needed

        # Feature engineering - ENSURE THESE MATCH YOUR TRAINING SCRIPT EXACTLY
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            df['bp_ratio'] = df['systolic_bp'] / df['diastolic_bp']
            df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
            df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['systolic_bp'] - df['diastolic_bp']) / 3

        if 'total_cholesterol' in df.columns and 'hdl' in df.columns:
            df['chol_ratio'] = df['total_cholesterol'] / df['hdl']
            df['cholesterol_hdl_ratio'] = df['total_cholesterol'] / df['hdl']


        if 'age' in df.columns and 'max_heart_rate' in df.columns:
            df['age_hr_interaction'] = df['age'] * df['max_heart_rate']
            df['rate_pressure_product'] = df['systolic_bp'] * df['max_heart_rate'] # Assuming systolic_bp is available

        if 'age' in df.columns and 'total_cholesterol' in df.columns:
            df['age_chol_interaction'] = df['age'] * df['total_cholesterol']

        if 'bmi' in df.columns and 'max_heart_rate' in df.columns:
            df['bmi_hr_interaction'] = df['bmi'] * df['max_heart_rate']

        if 'age' in df.columns:
            df['age_decile'] = (df['age'] // 10) * 10 # Example: 25 -> 20, 35 -> 30

        if 'bmi' in df.columns:
            # Simplified BMI categorization for numerical input
            df['bmi_category'] = 0 # Underweight
            df.loc[(df['bmi'] >= 18.5) & (df['bmi'] < 25), 'bmi_category'] = 1 # Normal weight
            df.loc[(df['bmi'] >= 25) & (df['bmi'] < 30), 'bmi_category'] = 2 # Overweight
            df.loc[df['bmi'] >= 30, 'bmi_category'] = 3 # Obese

        # Simple risk score (example, adjust based on your model's actual definition)
        df['simple_risk_score'] = (
            df['hypertension'] +
            df['diabetes_status'] +
            df['smoking_status'] +
            (df['family_history'] * 2) # Example: family history might be a stronger factor
        )

        return df

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return pd.DataFrame([input_data])


def apply_pca_transform(df, pca_model):
    """Apply PCA transformation if available and ensure correct feature names."""
    if pca_model is None:
        return df

    try:
        # Updated numerical_cols_for_pca to exactly 38 features, assuming 'simple_risk_score' was not used for PCA.
        # This list MUST match the exact 38 features your PCA model was trained on.
        numerical_cols_for_pca = [
            'age', 'sex', 'ethnicity', 'family_history', 'previous_cardiac_events',
            'diabetes_status', 'hypertension', 'systolic_bp', 'diastolic_bp',
            'resting_heart_rate', 'height_cm', 'weight_kg', 'bmi',
            'total_cholesterol', 'hdl', 'ldl', 'fasting_glucose', 'hba1c',
            'smoking_status', 'alcohol_per_week', 'physical_activity_hours',
            'diet_quality', 'stress_level', 'chest_pain_type', 'exercise_induced_angina',
            'resting_ecg', 'max_heart_rate', 'st_depression', 'st_slope',
            'num_vessels', 'thalassemia',
            'chol_ratio', 'bmi_category', 'age_decile', 'mean_arterial_pressure',
            'rate_pressure_product', 'age_chol_interaction', 'bmi_hr_interaction'
        ]

        # Filter the DataFrame to include only the columns expected by PCA
        # And ensure they are in the correct order
        df_for_pca = df[numerical_cols_for_pca].copy()

        # Apply PCA transformation
        pca_result = pca_model.transform(df_for_pca)

        # Create PCA feature dataframe, naming columns from pca_1 to pca_5 as expected
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'pca_{i+1}' for i in range(pca_result.shape[1])], # Naming pca_1, pca_2, etc.
            index=df.index
        )

        # Combine original features with PCA features.
        final_df = pd.concat([df, pca_df], axis=1)

        return final_df

    except Exception as e:
        print(f"Error in PCA transformation: {e}")
        print("Using original features without PCA (this might lead to model errors).")
        return df


def predict_single_sample(model_dict, input_data):
    """Make prediction for a single sample, ensuring feature names match the trained model."""
    try:
        if model_dict is None:
            raise ValueError("Model not loaded")

        model = model_dict['model']
        pca_model = model_dict.get('pca')

        # Preprocess the input data and apply engineered features
        df_processed = preprocess_input_data(input_data)

        # Apply PCA if available
        df_final_with_pca = apply_pca_transform(df_processed, pca_model)

        # Define the EXACT list of features the model was trained on, in the correct order.
        expected_feature_order = [
            'age', 'sex', 'ethnicity', 'family_history', 'previous_cardiac_events',
            'diabetes_status', 'hypertension', 'systolic_bp', 'diastolic_bp',
            'resting_heart_rate', 'height_cm', 'weight_kg', 'bmi',
            'total_cholesterol', 'hdl', 'ldl', 'fasting_glucose', 'hba1c',
            'smoking_status', 'alcohol_per_week', 'physical_activity_hours',
            'diet_quality', 'stress_level', 'chest_pain_type', 'exercise_induced_angina',
            'resting_ecg', 'max_heart_rate', 'st_depression', 'st_slope',
            'num_vessels', 'thalassemia',
            'chol_ratio', 'bmi_category', 'age_decile', 'mean_arterial_pressure',
            'rate_pressure_product', 'age_chol_interaction', 'bmi_hr_interaction',
            'simple_risk_score',
            'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5'
        ]

        # Create the final DataFrame for prediction, ensuring all expected columns are present
        # and in the correct order. Fill missing columns with 0.
        df_for_prediction = pd.DataFrame(columns=expected_feature_order)
        for col in expected_feature_order:
            if col in df_final_with_pca.columns:
                df_for_prediction[col] = df_final_with_pca[col]
            else:
                df_for_prediction[col] = 0  # Fill with 0 or a sensible default

        # Convert to appropriate types if necessary (e.g., int for boolean-like features)
        for col in ['sex', 'ethnicity', 'family_history', 'previous_cardiac_events', 'diabetes_status', 'hypertension',
                    'smoking_status', 'chest_pain_type', 'exercise_induced_angina', 'resting_ecg', 'st_slope', 'num_vessels',
                    'thalassemia', 'bmi_category', 'age_decile', 'simple_risk_score']:
             if col in df_for_prediction.columns:
                 df_for_prediction[col] = df_for_prediction[col].astype(int)


        print(f"Final input shape for model: {df_for_prediction.shape}")
        print(f"Final input columns for model: {df_for_prediction.columns.tolist()}")


        # Make prediction
        prediction = model.predict(df_for_prediction)[0]
        probability = model.predict_proba(df_for_prediction)[0]

        # Get probability for positive class (heart disease)
        prob_positive = probability[1] if len(probability) > 1 else probability[0]

        return int(prediction), float(prob_positive)

    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        raise


def format_prediction_results(prediction, probability):
    """Format prediction results for display"""
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
    if prediction == 1:
        interpretation = f"Based on the provided clinical data, the model predicts a {prob_percent:.1f}% probability of heart disease. This indicates a {risk_level.lower()} risk level. Immediate medical consultation is recommended for further evaluation and appropriate management."
    else:
        interpretation = f"Based on the provided clinical data, the model predicts a {prob_percent:.1f}% probability of heart disease. This indicates a {risk_level.lower()} risk level. Continue with regular preventive care and maintain a heart-healthy lifestyle."

    return {
        'prediction': prediction,
        'probability': prob_percent,
        'risk_level': risk_level,
        'interpretation': interpretation
    }


# Global model variable to avoid reloading
# Removed redundant 'main' blueprint definition as 'bp' is already defined at the top
# main = Blueprint('main', __name__)


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


def save_prediction_to_db(form_data, prediction_result, model_version="1.0"):
    """Save prediction data and results to database"""
    try:
        # Use current_app.app_context() to ensure proper Flask context
        with current_app.app_context():
            # Create new prediction record
            prediction = Prediction(
                # Demographics
                age=form_data['age'],
                sex=form_data['sex'],
                ethnicity=form_data['ethnicity'],

                # Medical History
                family_history=bool(form_data['family_history']),
                previous_cardiac_events=bool(form_data['previous_cardiac_events']),
                diabetes_status=bool(form_data['diabetes_status']),
                hypertension=bool(form_data['hypertension']),

                # Vital Signs
                systolic_bp=form_data['systolic_bp'],
                diastolic_bp=form_data['diastolic_bp'],
                resting_heart_rate=form_data['resting_heart_rate'],

                # Physical Measurements
                height_cm=form_data['height_cm'],
                weight_kg=form_data['weight_kg'],
                bmi=form_data['bmi'],

                # Laboratory Values
                total_cholesterol=form_data['total_cholesterol'],
                hdl=form_data['hdl'], # Changed from form.hdl.data to form_data['hdl']
                ldl=form_data['ldl'],
                fasting_glucose=form_data['fasting_glucose'],
                hba1c=form_data['hba1c'],

                # Lifestyle Factors
                smoking_status=form_data['smoking_status'],
                alcohol_per_week=form_data['alcohol_per_week'],
                physical_activity_hours=form_data['physical_activity_hours'],
                diet_quality=form_data['diet_quality'],
                stress_level=form_data['stress_level'],

                # Clinical Tests
                chest_pain_type=form_data['chest_pain_type'],
                exercise_induced_angina=bool(form_data['exercise_induced_angina']),
                resting_ecg=form_data['resting_ecg'],
                max_heart_rate=form_data['max_heart_rate'],
                st_depression=form_data['st_depression'],
                st_slope=form_data['st_slope'],
                num_vessels=form_data['num_vessels'],
                thalassemia=form_data['thalassemia'],

                # Prediction Results
                prediction_result=prediction_result['prediction'],
                probability=prediction_result['probability'] / 100,  # Convert percentage to decimal
                risk_level=prediction_result['risk_level'],
                interpretation=prediction_result['interpretation'],

                # Model Information
                model_version=model_version,
                model_features_used=json.dumps(list(form_data.keys())),

                # Session and request info
                session_id=session.get('session_id'),
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', '')[:500]
            )

            # Save to database
            db.session.add(prediction)
            db.session.commit()

            # Update daily summary
            update_daily_summary(prediction)

            return prediction.id

    except Exception as e:
        print(f"Error saving prediction to database: {e}")
        db.session.rollback()
        return None


def update_daily_summary(prediction):
    """Update or create daily prediction summary"""
    try:
        # Ensure operation within app context
        with current_app.app_context():
            today = date.today()
            summary = PredictionSummary.query.filter_by(date=today).first()

            if not summary:
                # Create new summary with default values
                summary = PredictionSummary(
                    date=today,
                    total_predictions=0,
                    positive_predictions=0,
                    negative_predictions=0,
                    avg_probability=0.0,
                    avg_age=0.0,
                    risk_very_low=0,
                    risk_low=0,
                    risk_moderate=0,
                    risk_high=0,
                    risk_very_high=0
                )

            # Update counts (handle None values)
            summary.total_predictions = (summary.total_predictions or 0) + 1

            if prediction.prediction_result == 1:
                summary.positive_predictions = (summary.positive_predictions or 0) + 1
            else:
                summary.negative_predictions = (summary.negative_predictions or 0) + 1

            # Update risk level counts (handle None values)
            risk_level = prediction.risk_level.lower().replace(' ', '_')
            if hasattr(summary, f'risk_{risk_level}'):
                current_count = getattr(summary, f'risk_{risk_level}') or 0  # Handle None
                setattr(summary, f'risk_{risk_level}', current_count + 1)

            # Recalculate averages
            today_predictions = Prediction.query.filter(
                func.date(Prediction.created_at) == today
            ).all()

            if today_predictions:
                summary.avg_probability = sum(p.probability for p in today_predictions) / len(today_predictions)
                summary.avg_age = sum(p.age for p in today_predictions) / len(today_predictions)

            db.session.add(summary)
            db.session.commit()

    except Exception as e:
        print(f"Error updating daily summary: {e}")
        db.session.rollback()


@bp.route('/')
def index():
    """Home page route with recent predictions summary"""
    try:
        # Get recent statistics
        total_predictions = Prediction.query.count()
        recent_predictions = Prediction.query.order_by(desc(Prediction.created_at)).limit(5).all()

        # Get today's summary
        today_summary = PredictionSummary.query.filter_by(date=date.today()).first()

        return render_template('index.html',
                               title='Heart Disease Prediction',
                               total_predictions=total_predictions,
                               recent_predictions=recent_predictions,
                               today_summary=today_summary)
    except Exception as e:
        print(f"Error loading index page: {e}")
        return render_template('index.html', title='Heart Disease Prediction')


@bp.route('/about')
def about():
    """About page route"""
    return render_template('about.html', title='About')


@bp.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction form route with database storage"""
    form = PredictionForm()

    if form.validate_on_submit():
        try:
            # Helper function to safely convert to int
            def safe_int(value, default=0):
                if value == '' or value is None:
                    return default
                return int(value)

            def safe_float(value, default=0.0):
                if value == '' or value is None:
                    return default
                return float(value)

            # Create input data dictionary with all required features
            input_data = {
                'age': form.age.data,
                'sex': safe_int(form.sex.data),
                'ethnicity': safe_int(form.ethnicity.data),
                'family_history': 1 if form.family_history.data else 0,
                'previous_cardiac_events': 1 if form.previous_cardiac_events.data else 0,
                'diabetes_status': 1 if form.diabetes_status.data else 0,
                'hypertension': 1 if form.hypertension.data else 0,
                'systolic_bp': form.systolic_bp.data,
                'diastolic_bp': form.diastolic_bp.data,
                'resting_heart_rate': form.resting_heart_rate.data,
                'height_cm': form.height_cm.data,
                'weight_kg': form.weight_kg.data,
                'bmi': form.weight_kg.data / ((form.height_cm.data / 100) ** 2),
                'total_cholesterol': form.total_cholesterol.data,
                'hdl': form.hdl.data,
                'ldl': form.ldl.data,
                'fasting_glucose': form.fasting_glucose.data,
                'hba1c': form.hba1c.data,
                'smoking_status': safe_int(form.smoking_status.data),
                'alcohol_per_week': form.alcohol_per_week.data,
                'physical_activity_hours': form.physical_activity_hours.data,
                'diet_quality': safe_int(form.diet_quality.data),
                'stress_level': safe_int(form.stress_level.data),
                'chest_pain_type': safe_int(form.chest_pain_type.data),
                'exercise_induced_angina': 1 if form.exercise_induced_angina.data else 0,
                'resting_ecg': safe_int(form.resting_ecg.data),
                'max_heart_rate': form.max_heart_rate.data,
                'st_depression': form.st_depression.data,
                'st_slope': safe_int(form.st_slope.data),
                'num_vessels': safe_int(form.num_vessels.data),
                'thalassemia': safe_int(form.thalassemia.data)
            }

            # Load model
            model = get_model()
            if model is None:
                flash('Model is currently unavailable. Please try again later.', 'error')
                return render_template('predict.html', title='Make Prediction', form=form)

            # Make prediction using the proper pipeline
            prediction, probability = predict_single_sample(model, input_data)
            result = format_prediction_results(prediction, probability)

            # Save prediction to database
            prediction_id = save_prediction_to_db(input_data, result)

            # Store prediction ID in session for later retrieval
            if prediction_id:
                session['last_prediction_id'] = prediction_id

            # Return results
            return render_template(
                'results.html',
                title='Prediction Results',
                prediction=result['prediction'],
                probability=result['probability'],  # This matches the template now
                risk_level=result['risk_level'],
                interpretation=result['interpretation'],
                form_data=input_data,
                prediction_id=prediction_id,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add timestamp
            )

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            flash(f'An error occurred during prediction: {str(e)}', 'error')
            return render_template('predict.html', title='Make Prediction', form=form)

    return render_template('predict.html', title='Make Prediction', form=form)


@bp.route('/predictions')
def predictions_history():
    """View prediction history"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 10

        predictions = Prediction.query.order_by(desc(Prediction.created_at)).paginate(
            page=page, per_page=per_page, error_out=False
        )

        return render_template('prediction_history.html',
                               title='Prediction History',
                               predictions=predictions)
    except Exception as e:
        print(f"Error loading predictions history: {e}")
        flash('Error loading prediction history.', 'error')
        return redirect(url_for('main.index'))


@bp.route('/prediction/<int:prediction_id>')
def view_prediction(prediction_id):
    """View detailed prediction"""
    try:
        prediction = Prediction.query.get_or_404(prediction_id)
        return render_template('view_prediction.html',
                               title=f'Prediction #{prediction_id}',
                               prediction=prediction)
    except Exception as e:
        print(f"Error loading prediction {prediction_id}: {e}")
        flash('Prediction not found.', 'error')
        return redirect(url_for('main.predictions_history'))


@bp.route('/analytics')
def analytics():
    """Analytics dashboard"""
    try:
        # Get overall statistics
        total_predictions = Prediction.query.count()
        positive_rate = db.session.query(func.avg(Prediction.prediction_result.cast(db.Float))).scalar() or 0

        # Get risk level distribution
        risk_distribution = db.session.query(
            Prediction.risk_level,
            func.count(Prediction.risk_level)
        ).group_by(Prediction.risk_level).all()

        # Get recent summaries
        recent_summaries = PredictionSummary.query.order_by(
            desc(PredictionSummary.date)
        ).limit(30).all()

        # Get age distribution
        age_distribution = db.session.query(
            func.case([
                (Prediction.age < 30, 'Under 30'),
                (Prediction.age < 50, '30-49'),
                (Prediction.age < 70, '50-69'),
            ], else_='70+').label('age_group'),
            func.count(Prediction.id)
        ).group_by('age_group').all()

        return render_template('analytics.html',
                               title='Analytics Dashboard',
                               total_predictions=total_predictions,
                               positive_rate=positive_rate * 100,
                               risk_distribution=risk_distribution,
                               recent_summaries=recent_summaries,
                               age_distribution=age_distribution)
    except Exception as e:
        print(f"Error loading analytics: {e}")
        flash('Error loading analytics dashboard.', 'error')
        return redirect(url_for('main.index'))


@bp.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions with database storage"""
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

        # Save to database
        prediction_id = save_prediction_to_db(data, result)

        # Add prediction ID to result
        result['prediction_id'] = prediction_id

        return jsonify(result)

    except Exception as e:
        print(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/predictions')
def api_predictions():
    """API endpoint to get predictions list"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        predictions = Prediction.query.order_by(desc(Prediction.created_at)).paginate(
            page=page, per_page=per_page, error_out=False
        )

        return jsonify({
            'predictions': [p.to_dict() for p in predictions.items],
            'total': predictions.total,
            'pages': predictions.pages,
            'current_page': page,
            'per_page': per_page
        })

    except Exception as e:
        print(f"API predictions error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/prediction/<int:prediction_id>')
def api_prediction_detail(prediction_id):
    """API endpoint to get detailed prediction"""
    try:
        prediction = Prediction.query.get_or_404(prediction_id)
        return jsonify(prediction.to_dict())
    except Exception as e:
        print(f"API prediction detail error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/analytics')
def api_analytics():
    """API endpoint for analytics data"""
    try:
        # Basic statistics
        total_predictions = Prediction.query.count()
        positive_predictions = Prediction.query.filter_by(prediction_result=1).count()

        # Risk level distribution
        risk_stats = db.session.query(
            Prediction.risk_level,
            func.count(Prediction.risk_level)
        ).group_by(Prediction.risk_level).all()

        # Recent activity (last 7 days)
        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_activity = db.session.query(
            func.date(Prediction.created_at).label('date'),
            func.count(Prediction.id).label('count')
        ).filter(
            Prediction.created_at >= week_ago
        ).group_by(func.date(Prediction.created_at)).all()

        return jsonify({
            'total_predictions': total_predictions,
            'positive_predictions': positive_predictions,
            'positive_rate': round((positive_predictions / total_predictions * 100) if total_predictions > 0 else 0, 2),
            'risk_distribution': dict(risk_stats),
            'recent_activity': [
                {'date': str(date), 'count': count}
                for date, count in recent_activity
            ]
        })

    except Exception as e:
        print(f"API analytics error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/health')
def health_check():
    """Health check endpoint with database status"""
    try:
        model = get_model()

        # Test database connection
        db_status = 'healthy'
        try:
            db.session.execute('SELECT 1')
        except:
            db_status = 'unhealthy'

        return jsonify({
            'status': 'healthy' if model is not None and db_status == 'healthy' else 'unhealthy',
            'model_loaded': model is not None,
            'database_status': db_status,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500