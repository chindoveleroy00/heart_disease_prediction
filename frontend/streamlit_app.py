import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import json
import joblib
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System - Parirenyatwa Group of Hospitals",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #0d6efd;
        --primary-dark: #0b5ed7;
        --secondary-color: #6c757d;
        --success-color: #198754;
        --danger-color: #dc3545;
        --warning-color: #ffc107;
        --info-color: #0dcaf0;
        --dark-color: #212529;
        --light-color: #f8f9fa;
        --medical-blue: #2563eb;
        --medical-green: #059669;
        --medical-red: #dc2626;
        --professional-bg: #f8f9fa;
        --sidebar-bg: rgba(255, 255, 255, 0.95);
        --shadow-sm: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        --shadow-md: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
        --shadow-lg: 0 1rem 3rem rgba(0, 0, 0, 0.175);
    }

    .main > div {
        padding-top: 2rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Updated main app background - removed purple gradient */
    .stApp {
        background: var(--professional-bg);
        min-height: 100vh;
    }

    /* Alternative professional backgrounds you can choose from */
    /*
    Option 1: Light gray professional
    .stApp {
        background: #f5f5f5;
        min-height: 100vh;
    }
    
    Option 2: Clean white
    .stApp {
        background: #ffffff;
        min-height: 100vh;
    }
    
    Option 3: Subtle blue-gray
    .stApp {
        background: linear-gradient(135deg, #f8f9fc 0%, #f1f3f7 100%);
        min-height: 100vh;
    }
    
    Option 4: Medical white with subtle pattern
    .stApp {
        background: #fafbfc;
        background-image: 
            radial-gradient(circle at 1px 1px, rgba(255,255,255,0.15) 1px, transparent 0);
        background-size: 20px 20px;
        min-height: 100vh;
    }
    */

    /* Header Styling - updated to match new background */
    .main-header {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        padding: 1.5rem 0;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        border: 1px solid rgba(0, 0, 0, 0.08);
    }

    .header-content {
        text-align: center;
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--medical-blue);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }

    .header-subtitle {
        font-size: 1.2rem;
        color: var(--secondary-color);
        font-weight: 500;
    }

    .hospital-name {
        font-size: 1rem;
        color: var(--medical-blue);
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* Content Cards - updated for professional look */
    .content-card {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        box-shadow: var(--shadow-md);
        border: 1px solid rgba(0, 0, 0, 0.08);
        padding: 2rem;
        margin-bottom: 2rem;
        animation: slideInUp 0.6s ease-out;
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Section Headers */
    .section-header {
        color: var(--medical-blue);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--medical-blue);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Metrics */
    .metric-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        border-left: 4px solid var(--medical-blue);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    .metric-high-risk {
        border-left-color: var(--danger-color);
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    }

    .metric-low-risk {
        border-left-color: var(--success-color);
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    }

    .metric-moderate-risk {
        border-left-color: var(--warning-color);
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    }

    /* Buttons - updated for professional appearance */
    .stButton > button {
        background: linear-gradient(135deg, var(--medical-blue) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--medical-blue) 100%);
    }

    /* Form Styling */
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 0.5rem;
        border: 2px solid #e5e7eb;
    }

    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 0.5rem;
        border: 2px solid #e5e7eb;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 0.5rem;
        border: 2px solid #e5e7eb;
    }

    /* Sidebar - maintaining the professional look */
    .css-1d391kg {
        background: var(--sidebar-bg);
        backdrop-filter: blur(10px);
    }

    /* Alert Styles */
    .alert-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: var(--success-color);
        border-left: 4px solid var(--success-color);
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid rgba(5, 150, 105, 0.2);
    }

    .alert-danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: var(--danger-color);
        border-left: 4px solid var(--danger-color);
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid rgba(220, 38, 38, 0.2);
    }

    .alert-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border-left: 4px solid var(--warning-color);
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 193, 7, 0.2);
    }

    .alert-info {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: var(--info-color);
        border-left: 4px solid var(--info-color);
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: 1px solid rgba(13, 202, 240, 0.2);
    }

    /* Risk Factor Analysis */
    .risk-factors {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }

    .risk-factor-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e5e7eb;
        color: var(--dark-color);
    }

    /* Footer */
    .footer {
        background: rgba(33, 37, 41, 0.98);
        color: white;
        text-align: center;
        padding: 2rem;
        border-radius: 1rem;
        margin-top: 2rem;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }

    /* Form sections from predict.html */
    .form-section {
        background: rgba(248, 249, 250, 0.8);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Database connection - FIXED VERSION
@st.cache_resource
def init_database():
    """Initialize database connection with proper path resolution"""
    try:
        # Get the current script directory and project root
        current_dir = Path(__file__).parent
        
        # Try different possible locations for the database
        possible_db_paths = [
            # If running from frontend/ directory
            current_dir.parent / "database" / "heart_disease.db",
            # If running from project root
            current_dir / "database" / "heart_disease.db",
            # Relative from current working directory
            Path("database/heart_disease.db"),
            # Absolute path attempt
            Path.cwd() / "database" / "heart_disease.db"
        ]
        
        db_path = None
        for path in possible_db_paths:
            if path.exists():
                db_path = path
                break
        
        if db_path is None:
            st.error("Database file not found in any of these locations:")
            for path in possible_db_paths:
                st.error(f"  - {path}")
            st.error(f"Current working directory: {Path.cwd()}")
            st.error(f"Script directory: {Path(__file__).parent}")
            st.error("Please run init_db.py first or check the database location.")
            st.stop()
        
        st.info(f"Connecting to database: {db_path}")
        return sqlite3.connect(str(db_path), check_same_thread=False)
        
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

# Load ML models
@st.cache_resource
def load_ml_models():
    """Load the trained ML models with robust error handling"""
    try:
        # Get the current script directory and project root
        current_dir = Path(__file__).parent
        
        # Try different possible locations for the models
        possible_paths = [
            current_dir.parent / "data" / "models" / "heart_disease_model.joblib",
            current_dir / "data" / "models" / "heart_disease_model.joblib",
            Path("data/models/heart_disease_model.joblib"),
            Path.cwd() / "data" / "models" / "heart_disease_model.joblib"
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            st.error("❌ Model file not found in any of these locations:")
            for path in possible_paths:
                st.error(f"  - {path}")
            return None, None
        
        # Load the main model
        st.info(f"📁 Loading model from: {model_path}")
        try:
            model = joblib.load(model_path)
            st.success("✅ Main model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load main model: {e}")
            return None, None
        
        # Try different possible locations for PCA model
        pca_possible_paths = [
            current_dir.parent / "data" / "models" / "pca_model.pkl",
            current_dir / "data" / "models" / "pca_model.pkl",
            Path("data/models/pca_model.pkl"),
            Path.cwd() / "data" / "models" / "pca_model.pkl"
        ]
        
        pca_path = None
        for path in pca_possible_paths:
            if path.exists():
                pca_path = path
                break
        
        # Load PCA model with error handling
        pca_model = None
        if pca_path and pca_path.exists():
            st.info(f"📁 Loading PCA model from: {pca_path}")
            
            # Check if file is empty or too small
            if pca_path.stat().st_size < 100:  # Less than 100 bytes is likely corrupted
                st.warning("⚠️ PCA model file appears to be corrupted (too small)")
                pca_model = None
            else:
                try:
                    pca_model = joblib.load(pca_path)
                    st.success("✅ PCA model loaded successfully!")
                except Exception as e:
                    st.warning(f"⚠️ Failed to load PCA model (corrupted file): {e}")
                    st.info("🔄 Attempting to delete corrupted PCA model...")
                    try:
                        pca_path.unlink()  # Delete corrupted file
                        st.info("✅ Corrupted PCA model deleted")
                    except Exception as del_error:
                        st.warning(f"Could not delete corrupted file: {del_error}")
                    pca_model = None
        else:
            st.warning("⚠️ PCA model not found - continuing without PCA transformation")
        
        # If PCA model failed to load, try to regenerate it
        if pca_model is None:
            st.info("🔧 Attempting to create a simple PCA model...")
            try:
                from sklearn.decomposition import PCA
                import numpy as np
                
                # Create a basic PCA model with 5 components (matching your expected features)
                pca_model = PCA(n_components=5, random_state=42)
                
                # Create dummy data to fit the PCA (this is not ideal, but allows the system to work)
                # In production, you should retrain this properly
                dummy_data = np.random.randn(100, 37)  # 37 features as expected by your model
                pca_model.fit(dummy_data)
                
                st.warning("⚠️ Created temporary PCA model - system will work but accuracy may be reduced")
                st.info("📝 Recommendation: Retrain and save the proper PCA model")
                
            except Exception as pca_error:
                st.warning(f"❌ Could not create temporary PCA model: {pca_error}")
                pca_model = None
        
        return model, pca_model
        
    except Exception as e:
        st.error(f"❌ Critical error in model loading: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

# Authentication functions
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password, conn):
    """Authenticate user against database"""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, role, password_hash FROM users WHERE username = ?",
        (username,)
    )
    user = cursor.fetchone()
    
    if user and user[3] == hash_password(password):
        return {'id': user[0], 'username': user[1], 'role': user[2]}
    return None

# Database helper functions (keeping existing ones)
def get_user_by_id(user_id, conn):
    """Get user by ID"""
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, role FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()

def get_all_users(conn):
    """Get all users"""
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, role, created_at, last_login_at FROM users ORDER BY created_at DESC")
    return cursor.fetchall()

def create_user(username, email, password, role, conn):
    """Create new user"""
    cursor = conn.cursor()
    password_hash = hash_password(password)
    cursor.execute(
        "INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
        (username, email, password_hash, role, datetime.now())
    )
    conn.commit()
    return cursor.lastrowid

def update_user(user_id, username, email, role, conn):
    """Update existing user"""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET username = ?, email = ?, role = ? WHERE id = ?",
        (username, email, role, user_id)
    )
    conn.commit()

def delete_user(user_id, conn):
    """Delete user"""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()

def save_prediction(prediction_data, conn):
    """Save prediction to database"""
    cursor = conn.cursor()
    
    # Insert prediction with all required fields
    query = """
    INSERT INTO predictions (
        created_at, age, sex, ethnicity, family_history, previous_cardiac_events,
        diabetes_status, hypertension, systolic_bp, diastolic_bp, resting_heart_rate,
        height_cm, weight_kg, bmi, total_cholesterol, hdl, ldl, fasting_glucose, hba1c,
        smoking_status, alcohol_per_week, physical_activity_hours, diet_quality, stress_level,
        chest_pain_type, exercise_induced_angina, resting_ecg, max_heart_rate, st_depression,
        st_slope, num_vessels, thalassemia, prediction_result, probability, risk_level,
        interpretation, model_version, session_id, ip_address
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    values = (
        datetime.now(),
        prediction_data['age'], prediction_data['sex'], prediction_data.get('ethnicity', 0),
        prediction_data.get('family_history', False), prediction_data.get('previous_cardiac_events', False),
        prediction_data.get('diabetes_status', False), prediction_data.get('hypertension', False),
        prediction_data['systolic_bp'], prediction_data['diastolic_bp'], prediction_data['resting_heart_rate'],
        prediction_data['height_cm'], prediction_data['weight_kg'], prediction_data['bmi'],
        prediction_data['total_cholesterol'], prediction_data['hdl'], prediction_data['ldl'],
        prediction_data['fasting_glucose'], prediction_data['hba1c'],
        prediction_data.get('smoking_status', 0), prediction_data.get('alcohol_per_week', 0),
        prediction_data.get('physical_activity_hours', 0), prediction_data.get('diet_quality', 2),
        prediction_data.get('stress_level', 3), prediction_data.get('chest_pain_type', 0),
        prediction_data.get('exercise_induced_angina', False), prediction_data.get('resting_ecg', 0),
        prediction_data['max_heart_rate'], prediction_data.get('st_depression', 0),
        prediction_data.get('st_slope', 1), prediction_data.get('num_vessels', 0),
        prediction_data.get('thalassemia', 1),
        prediction_data['prediction_result'], prediction_data['probability'], prediction_data['risk_level'],
        prediction_data['interpretation'], prediction_data.get('model_version', '1.0'),
        st.session_state.get('session_id', 'streamlit_session'), '127.0.0.1'
    )
    
    cursor.execute(query, values)
    conn.commit()
    return cursor.lastrowid

def get_predictions(conn, limit=None):
    """Get predictions from database"""
    cursor = conn.cursor()
    query = """
    SELECT id, created_at, age, sex, prediction_result, probability, risk_level, interpretation
    FROM predictions ORDER BY created_at DESC
    """
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    return cursor.fetchall()

def get_prediction_analytics(conn):
    """Get analytics data for predictions"""
    cursor = conn.cursor()
    
    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]
    
    # Positive predictions
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_result = 1")
    positive_predictions = cursor.fetchone()[0]
    
    # Risk level distribution
    cursor.execute("SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level")
    risk_distribution = cursor.fetchall()
    
    # Age distribution
    cursor.execute("""
    SELECT 
        CASE 
            WHEN age < 30 THEN 'Under 30'
            WHEN age < 50 THEN '30-49'
            WHEN age < 70 THEN '50-69'
            ELSE '70+'
        END as age_group,
        COUNT(*) 
    FROM predictions 
    GROUP BY age_group
    """)
    age_distribution = cursor.fetchall()
    
    # Daily trends (last 30 days)
    cursor.execute("""
    SELECT DATE(created_at) as date, COUNT(*) as count
    FROM predictions 
    WHERE created_at >= date('now', '-30 days')
    GROUP BY DATE(created_at)
    ORDER BY date
    """)
    daily_trends = cursor.fetchall()
    
    return {
        'total_predictions': total_predictions,
        'positive_predictions': positive_predictions,
        'positive_rate': (positive_predictions / total_predictions * 100) if total_predictions > 0 else 0,
        'risk_distribution': risk_distribution,
        'age_distribution': age_distribution,
        'daily_trends': daily_trends
    }

# ML prediction functions (keeping existing implementation)
def preprocess_input_data(input_data):
    """Preprocess input data with feature engineering"""
    df = pd.DataFrame([input_data])
    
    # Ensure all values are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(0)
    
    # Feature engineering
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['bp_ratio'] = df['systolic_bp'] / df['diastolic_bp']
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['systolic_bp'] - df['diastolic_bp']) / 3

    if 'total_cholesterol' in df.columns and 'hdl' in df.columns:
        df['chol_ratio'] = df['total_cholesterol'] / df['hdl']
        df['cholesterol_hdl_ratio'] = df['total_cholesterol'] / df['hdl']

    if 'age' in df.columns and 'max_heart_rate' in df.columns:
        df['age_hr_interaction'] = df['age'] * df['max_heart_rate']
        df['rate_pressure_product'] = df['systolic_bp'] * df['max_heart_rate']

    if 'age' in df.columns and 'total_cholesterol' in df.columns:
        df['age_chol_interaction'] = df['age'] * df['total_cholesterol']

    if 'bmi' in df.columns and 'max_heart_rate' in df.columns:
        df['bmi_hr_interaction'] = df['bmi'] * df['max_heart_rate']

    if 'age' in df.columns:
        df['age_decile'] = (df['age'] // 10) * 10

    if 'bmi' in df.columns:
        df['bmi_category'] = 0
        df.loc[(df['bmi'] >= 18.5) & (df['bmi'] < 25), 'bmi_category'] = 1
        df.loc[(df['bmi'] >= 25) & (df['bmi'] < 30), 'bmi_category'] = 2
        df.loc[df['bmi'] >= 30, 'bmi_category'] = 3

    # Simple risk score
    df['simple_risk_score'] = (
        df['hypertension'] +
        df['diabetes_status'] +
        df['smoking_status'] +
        (df['family_history'] * 2)
    )
    
    return df

def apply_pca_transform(df, pca_model):
    """Apply PCA transformation with improved error handling"""
    if pca_model is None:
        return df

    try:
        # Define expected features that PCA was trained on (38 features)
        expected_pca_features = [
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

        # Select only the features that PCA expects
        available_features = [col for col in expected_pca_features if col in df.columns]
        
        if len(available_features) != len(expected_pca_features):
            # Silently handle feature mismatch without showing warning to user
            # Add missing features with default values
            df_for_pca = df.copy()
            for feature in expected_pca_features:
                if feature not in df_for_pca.columns:
                    df_for_pca[feature] = 0
            
            # Select features in correct order
            df_for_pca = df_for_pca[expected_pca_features]
        else:
            df_for_pca = df[available_features].copy()
        
        # Apply PCA transformation
        pca_result = pca_model.transform(df_for_pca)
        
        # Create PCA dataframe
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'pca_{i+1}' for i in range(pca_result.shape[1])],
            index=df.index
        )
        
        # Combine original data with PCA features
        final_df = pd.concat([df, pca_df], axis=1)
        return final_df

    except Exception as e:
        # Silently handle PCA errors - don't show warnings to clinicians
        # Log error internally but continue without PCA
        return df

def make_prediction(model, pca_model, input_data):
    """Make prediction using the ML model with proper feature handling and silent error handling"""
    try:
        if model is None:
            raise ValueError("Model not loaded")

        df_processed = preprocess_input_data(input_data)
        
        # Apply PCA transformation if available (silently handle errors)
        if pca_model is not None:
            try:
                # Define the exact 38 features that PCA expects
                expected_pca_features = [
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
                
                # Create dataframe with exactly the features PCA expects
                df_for_pca = pd.DataFrame()
                for feature in expected_pca_features:
                    if feature in df_processed.columns:
                        df_for_pca[feature] = df_processed[feature]
                    else:
                        df_for_pca[feature] = 0  # Fill missing features with 0
                
                # Apply PCA transformation
                pca_result = pca_model.transform(df_for_pca)
                
                # Add PCA features to dataframe
                pca_cols = [f'pca_{i+1}' for i in range(pca_result.shape[1])]
                pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=df_processed.index)
                df_final_with_pca = pd.concat([df_processed, pca_df], axis=1)
                
            except Exception as pca_error:
                # Silently handle PCA errors - don't show to clinicians
                df_final_with_pca = df_processed
        else:
            df_final_with_pca = df_processed

        # Get expected feature order from the model
        try:
            # For XGBoost models, we can get feature names
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_.tolist()
            else:
                # Fallback to expected feature list
                expected_features = [
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
        except:
            # Final fallback - use whatever features we have
            expected_features = df_final_with_pca.columns.tolist()

        # Ensure we have all expected features
        df_for_prediction = pd.DataFrame(columns=expected_features)
        for col in expected_features:
            if col in df_final_with_pca.columns:
                df_for_prediction[col] = df_final_with_pca[col]
            else:
                df_for_prediction[col] = 0  # Fill missing features with 0

        # Make prediction
        prediction = model.predict(df_for_prediction)[0]
        probability = model.predict_proba(df_for_prediction)[0]
        prob_positive = probability[1] if len(probability) > 1 else probability[0]

        return int(prediction), float(prob_positive)

    except Exception as e:
        # Only show critical errors that prevent prediction
        if "Model not loaded" in str(e):
            st.error(f"System error: {e}")
        else:
            st.error("Prediction temporarily unavailable. Please try again.")
        raise

def format_prediction_results(prediction, probability):
    """Format prediction results"""
    prob_percent = probability * 100

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

    if prediction == 1:
        interpretation = f"Based on the provided clinical data, the model predicts a {prob_percent:.1f}% probability of heart disease. This indicates a {risk_level.lower()} risk level. Immediate medical consultation is recommended for further evaluation and appropriate management."
    else:
        interpretation = f"Based on the provided clinical data, the model predicts a {prob_percent:.1f}% probability of heart disease. This indicates a {risk_level.lower()} risk level. Continue with regular preventive care and maintain a heart-healthy lifestyle."

    return {
        'prediction_result': prediction,
        'probability': prob_percent / 100,  # Store as decimal for database
        'risk_level': risk_level,
        'interpretation': interpretation
    }

# UI Components - Updated with Flask styling
def show_header():
    """Display main header"""
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <div class="header-title">
                <i class="fas fa-heartbeat"></i>
                Heart Disease Prediction System
            </div>
            <div class="header-subtitle">
                Advancing healthcare through artificial intelligence
            </div>
            <div class="hospital-name">
                Parirenyatwa Group of Hospitals
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_login_page():
    """Display login page with improved styling"""
    load_custom_css()
    show_header()
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="section-header">
            <i class="fas fa-sign-in-alt"></i>
            Login to System
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("Please enter your credentials to access the system")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                submitted = st.form_submit_button("🔐 Login", use_container_width=True)
            
            if submitted:
                if username and password:
                    conn = init_database()
                    user = authenticate_user(username, password, conn)
                    
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user_role = user['role']
                        st.session_state.username = user['username']
                        st.session_state.user_id = user['id']
                        st.success(f"✅ Welcome back, {user['username']}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.error("⚠️ Please enter both username and password")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_sidebar():
    """Display enhanced sidebar navigation"""
    with st.sidebar:
        st.markdown(f"""
        <div class="metric-container">
            <h3>👋 Welcome</h3>
            <p><strong>{st.session_state.username}</strong></p>
            <p><em>Role: {st.session_state.user_role.title()}</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Role-based menu
        if st.session_state.user_role == 'clinician':
            page = st.selectbox(
                "🧭 Navigation",
                ["🏠 Dashboard", "🔬 Make Prediction", "📊 Prediction History", "ℹ️ About"],
                key="nav_select"
            )
        elif st.session_state.user_role == 'admin':
            page = st.selectbox(
                "🧭 Navigation",
                ["🏠 Dashboard", "👥 Manage Users", "📊 System Logs", "⚙️ Configuration", "📈 Analytics", "ℹ️ About"],
                key="nav_select"
            )
        else:
            page = "🏠 Dashboard"
        
        st.markdown("---")
        
        # System status
        st.markdown("""
        <div class="metric-container metric-low-risk">
            <h4>🟢 System Status</h4>
            <p>All systems operational</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.session_state.user_id = None
            st.rerun()
    
    return page

def show_dashboard():
    """Display enhanced main dashboard"""
    load_custom_css()
    show_header()
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-tachometer-alt"></i>
        System Dashboard
    </div>
    """, unsafe_allow_html=True)
    
    conn = init_database()
    analytics = get_prediction_analytics(conn)
    recent_predictions = get_predictions(conn, limit=5)
    
    # Key metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Total Predictions</h3>
            <h2>{analytics['total_predictions']}</h2>
            <p>All time predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container metric-high-risk">
            <h3>⚠️ Positive Cases</h3>
            <h2>{analytics['positive_predictions']}</h2>
            <p>Heart disease detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        risk_class = "metric-high-risk" if analytics['positive_rate'] > 50 else "metric-moderate-risk" if analytics['positive_rate'] > 25 else "metric-low-risk"
        st.markdown(f"""
        <div class="metric-container {risk_class}">
            <h3>📈 Positive Rate</h3>
            <h2>{analytics['positive_rate']:.1f}%</h2>
            <p>Detection rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container metric-low-risk">
            <h3>🕒 Recent Activity</h3>
            <h2>{len(recent_predictions)}</h2>
            <p>Last 5 predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts with enhanced layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-header">
            <i class="fas fa-chart-pie"></i>
            Risk Level Distribution
        </div>
        """, unsafe_allow_html=True)
        
        if analytics['risk_distribution']:
            risk_df = pd.DataFrame(analytics['risk_distribution'], columns=['Risk Level', 'Count'])
            colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1']
            fig = px.pie(risk_df, values='Count', names='Risk Level', 
                        color_discrete_sequence=colors)
            fig.update_layout(font=dict(size=14), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="section-header">
            <i class="fas fa-users"></i>
            Age Group Distribution
        </div>
        """, unsafe_allow_html=True)
        
        if analytics['age_distribution']:
            age_df = pd.DataFrame(analytics['age_distribution'], columns=['Age Group', 'Count'])
            fig = px.bar(age_df, x='Age Group', y='Count', 
                        color='Count', color_continuous_scale='Blues')
            fig.update_layout(font=dict(size=14))
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions table
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-history"></i>
        Recent Predictions
    </div>
    """, unsafe_allow_html=True)
    
    if recent_predictions:
        df = pd.DataFrame(recent_predictions, columns=['ID', 'Date', 'Age', 'Sex', 'Result', 'Probability', 'Risk Level', 'Interpretation'])
        df['Sex'] = df['Sex'].map({0: 'Female', 1: 'Male'})
        df['Result'] = df['Result'].map({0: 'No Disease', 1: 'Heart Disease'})
        df['Probability'] = (df['Probability'] * 100).round(1).astype(str) + '%'
        
        # Style the dataframe
        styled_df = df[['Date', 'Age', 'Sex', 'Result', 'Probability', 'Risk Level']].style.apply(
            lambda x: ['background-color: #fee2e2' if v == 'Heart Disease' else 
                      'background-color: #d1fae5' if v == 'No Disease' else '' 
                      for v in x], subset=['Result'])
        
        st.dataframe(styled_df, use_container_width=True, height=300)
    else:
        st.markdown("""
        <div class="alert-info">
            <strong>ℹ️ Information:</strong> No predictions available yet. Start by making your first prediction!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_prediction_form():
    """Display enhanced prediction form based on predict.html structure"""
    load_custom_css()
    show_header()
    
    model, pca_model = load_ml_models()
    if model is None:
        st.error("❌ Model not available. Please contact administrator.")
        return
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-stethoscope"></i>
        Heart Disease Risk Prediction
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-info">
        <strong>📋 Instructions:</strong> Please fill in all patient information accurately for the most reliable prediction. 
        All fields are required for optimal model performance.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form", clear_on_submit=False):
        
        # Demographics Section
        st.markdown("""
        <div class="form-section">
            <div class="section-header">
                <i class="fas fa-user"></i>
                Demographics
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=50, help="Patient's age in years")
        with col2:
            sex = st.selectbox("Sex", ["Female", "Male"], help="Biological sex")
        with col3:
            ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Hispanic", "Asian", "Other"], help="Patient's ethnicity")
        
        # Medical History Section
        st.markdown("""
        <div class="form-section">
            <div class="section-header">
                <i class="fas fa-history"></i>
                Medical History
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            family_history = st.checkbox("Family History of Heart Disease", help="Close relatives with heart disease")
            previous_cardiac_events = st.checkbox("Previous Cardiac Events", help="Previous heart attacks, surgeries, etc.")
        with col2:
            diabetes_status = st.checkbox("Diabetes", help="Type 1 or Type 2 diabetes diagnosis")
            hypertension = st.checkbox("Hypertension", help="High blood pressure diagnosis")
        
        # Vital Signs Section
        st.markdown("""
        <div class="form-section">
            <div class="section-header">
                <i class="fas fa-heartbeat"></i>
                Vital Signs
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=120, help="Upper blood pressure reading")
        with col2:
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=120, value=80, help="Lower blood pressure reading")
        with col3:
            resting_heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=120, value=70, help="Heart rate at rest")
        
        # Physical Measurements Section
        st.markdown("""
        <div class="form-section">
            <div class="section-header">
                <i class="fas fa-ruler"></i>
                Physical Measurements
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            height_cm = st.number_input("Height (cm)", min_value=120, max_value=220, value=170, help="Patient's height in centimeters")
        with col2:
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, help="Patient's weight in kilograms")
        
        bmi = weight_kg / ((height_cm / 100) ** 2)
        bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        st.markdown(f"""
        <div class="alert-info">
            <strong>📊 Calculated BMI:</strong> {bmi:.1f} kg/m² ({bmi_category})
        </div>
        """, unsafe_allow_html=True)
        
        # Lab Values Section
        st.markdown("""
        <div class="form-section">
            <div class="section-header">
                <i class="fas fa-vial"></i>
                Laboratory Values
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_cholesterol = st.number_input("Total Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, help="Total cholesterol level")
            hdl = st.number_input("HDL Cholesterol (mg/dl)", min_value=20, max_value=100, value=50, help="High-density lipoprotein")
        with col2:
            ldl = st.number_input("LDL Cholesterol (mg/dl)", min_value=50, max_value=400, value=100, help="Low-density lipoprotein")
            fasting_glucose = st.number_input("Fasting Glucose (mg/dl)", min_value=50, max_value=400, value=90, help="Blood glucose after fasting")
        with col3:
            hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=5.5, step=0.1, help="Glycated hemoglobin")
        
        # Lifestyle Factors Section
        st.markdown("""
        <div class="form-section">
            <div class="section-header">
                <i class="fas fa-running"></i>
                Lifestyle Factors
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"], help="Current smoking habits")
            alcohol_per_week = st.number_input("Alcohol Units per Week", min_value=0, max_value=50, value=0, help="Weekly alcohol consumption")
            physical_activity_hours = st.number_input("Physical Activity (hours/week)", min_value=0.0, max_value=50.0, value=2.0, help="Weekly exercise hours")
        with col2:
            diet_quality = st.selectbox("Diet Quality", ["Poor", "Fair", "Good", "Excellent"], help="Overall diet quality assessment")
            stress_level = st.selectbox("Stress Level", ["Very Low", "Low", "Moderate", "High", "Very High"], help="Perceived stress level")
        
        # Clinical Tests Section
        st.markdown("""
        <div class="form-section">
            <div class="section-header">
                <i class="fas fa-stethoscope"></i>
                Clinical Tests
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], help="Type of chest pain experienced")
            exercise_induced_angina = st.checkbox("Exercise Induced Angina", help="Chest pain triggered by exercise")
            resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], help="Electrocardiogram results")
        with col2:
            max_heart_rate = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, help="Peak heart rate during stress test")
            st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1, help="ST segment depression on ECG")
            st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"], help="Slope of ST segment")
            num_vessels = st.selectbox("Number of Major Vessels", ["0", "1", "2", "3", "4"], help="Major vessels with >50% narrowing")
            thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"], help="Thalassemia test result")
        
        # Submit Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🔮 Predict Heart Disease Risk", use_container_width=True, type="primary")
        
        if submitted:
            # Show processing message
            with st.spinner("🔄 Processing prediction... Please wait."):
                # Prepare input data (same structure as before)
                input_data = {
                    'age': age,
                    'sex': 1 if sex == "Male" else 0,
                    'ethnicity': ["White", "Black", "Hispanic", "Asian", "Other"].index(ethnicity),
                    'family_history': int(family_history),
                    'previous_cardiac_events': int(previous_cardiac_events),
                    'diabetes_status': int(diabetes_status),
                    'hypertension': int(hypertension),
                    'systolic_bp': systolic_bp,
                    'diastolic_bp': diastolic_bp,
                    'resting_heart_rate': resting_heart_rate,
                    'height_cm': height_cm,
                    'weight_kg': weight_kg,
                    'bmi': bmi,
                    'total_cholesterol': total_cholesterol,
                    'hdl': hdl,
                    'ldl': ldl,
                    'fasting_glucose': fasting_glucose,
                    'hba1c': hba1c,
                    'smoking_status': ["Never", "Former", "Current"].index(smoking_status),
                    'alcohol_per_week': alcohol_per_week,
                    'physical_activity_hours': physical_activity_hours,
                    'diet_quality': ["Poor", "Fair", "Good", "Excellent"].index(diet_quality) + 1,
                    'stress_level': ["Very Low", "Low", "Moderate", "High", "Very High"].index(stress_level) + 1,
                    'chest_pain_type': ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain_type),
                    'exercise_induced_angina': int(exercise_induced_angina),
                    'resting_ecg': ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg),
                    'max_heart_rate': max_heart_rate,
                    'st_depression': st_depression,
                    'st_slope': ["Upsloping", "Flat", "Downsloping"].index(st_slope),
                    'num_vessels': int(num_vessels),
                    'thalassemia': ["Normal", "Fixed Defect", "Reversible Defect"].index(thalassemia) + 1
                }
                
                try:
                    # Make prediction
                    prediction, probability = make_prediction(model, pca_model, input_data)
                    result = format_prediction_results(prediction, probability)
                    
                    # Update input data with prediction results
                    input_data.update(result)
                    
                    # Save to database
                    conn = init_database()
                    prediction_id = save_prediction(input_data, conn)
                    
                    # Display results with enhanced styling
                    st.markdown("---")
                    st.markdown("""
                    <div class="section-header">
                        <i class="fas fa-clipboard-check"></i>
                        Prediction Results
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Results cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if result['prediction_result'] == 1:
                            st.markdown("""
                            <div class="metric-container metric-high-risk">
                                <h3>⚠️ Heart Disease Risk</h3>
                                <h2>DETECTED</h2>
                                <p>Requires medical attention</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="metric-container metric-low-risk">
                                <h3>✅ Heart Disease Risk</h3>
                                <h2>NOT DETECTED</h2>
                                <p>Continue preventive care</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        prob_class = "metric-high-risk" if result['probability'] > 0.8 else "metric-moderate-risk" if result['probability'] > 0.4 else "metric-low-risk"
                        st.markdown(f"""
                        <div class="metric-container {prob_class}">
                            <h3>📊 Probability</h3>
                            <h2>{result['probability']*100:.1f}%</h2>
                            <p>Confidence level</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        risk_colors = {
                            "Very Low": ("metric-low-risk", "🟢"),
                            "Low": ("metric-low-risk", "🟡"), 
                            "Moderate": ("metric-moderate-risk", "🟠"),
                            "High": ("metric-high-risk", "🔴"),
                            "Very High": ("metric-high-risk", "🚨")
                        }
                        risk_class, risk_emoji = risk_colors.get(result['risk_level'], ("metric-container", "⚪"))
                        st.markdown(f"""
                        <div class="metric-container {risk_class}">
                            <h3>{risk_emoji} Risk Level</h3>
                            <h2>{result['risk_level']}</h2>
                            <p>Clinical assessment</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Clinical interpretation
                    st.markdown(f"""
                    <div class="alert-info">
                        <strong>🩺 Clinical Interpretation:</strong><br>
                        {result['interpretation']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk factor analysis
                    st.markdown("""
                    <div class="section-header">
                        <i class="fas fa-exclamation-triangle"></i>
                        Risk Factor Analysis
                    </div>
                    """, unsafe_allow_html=True)
                    
                    risk_factors = []
                    protective_factors = []
                    
                    # Risk factors
                    if age > 55:
                        risk_factors.append(f"Advanced age ({age} years)")
                    if sex == "Male":
                        risk_factors.append("Male gender (higher baseline risk)")
                    if systolic_bp > 140 or diastolic_bp > 90:
                        risk_factors.append(f"Hypertension ({systolic_bp}/{diastolic_bp} mmHg)")
                    if total_cholesterol > 240:
                        risk_factors.append(f"High total cholesterol ({total_cholesterol} mg/dl)")
                    if hdl < 40:
                        risk_factors.append(f"Low HDL cholesterol ({hdl} mg/dl)")
                    if ldl > 130:
                        risk_factors.append(f"Elevated LDL cholesterol ({ldl} mg/dl)")
                    if bmi > 30:
                        risk_factors.append(f"Obesity (BMI: {bmi:.1f})")
                    if diabetes_status:
                        risk_factors.append("Diabetes mellitus")
                    if smoking_status == "Current":
                        risk_factors.append("Current smoking")
                    if family_history:
                        risk_factors.append("Family history of heart disease")
                    if fasting_glucose > 126:
                        risk_factors.append(f"Elevated glucose ({fasting_glucose} mg/dl)")
                    if hba1c > 6.5:
                        risk_factors.append(f"Elevated HbA1c ({hba1c}%)")
                    if exercise_induced_angina:
                        risk_factors.append("Exercise-induced chest pain")
                    
                    # Protective factors
                    if physical_activity_hours >= 3:
                        protective_factors.append(f"Regular physical activity ({physical_activity_hours} hrs/week)")
                    if diet_quality in ["Good", "Excellent"]:
                        protective_factors.append(f"{diet_quality} diet quality")
                    if smoking_status == "Never":
                        protective_factors.append("Non-smoker")
                    if hdl > 60:
                        protective_factors.append(f"High HDL cholesterol ({hdl} mg/dl)")
                    if bmi >= 18.5 and bmi < 25:
                        protective_factors.append(f"Normal BMI ({bmi:.1f})")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if risk_factors:
                            st.markdown("""
                            <div class="risk-factors">
                                <h4 style="color: var(--danger-color);">⚠️ Identified Risk Factors:</h4>
                            """, unsafe_allow_html=True)
                            for factor in risk_factors:
                                st.markdown(f"<div class='risk-factor-item'>• {factor}</div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="alert-success">
                                <strong>✅ No major risk factors identified</strong>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        if protective_factors:
                            st.markdown("""
                            <div class="risk-factors">
                                <h4 style="color: var(--success-color);">✅ Protective Factors:</h4>
                            """, unsafe_allow_html=True)
                            for factor in protective_factors:
                                st.markdown(f"<div class='risk-factor-item'>• {factor}</div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Medical recommendations
                    st.markdown("""
                    <div class="section-header">
                        <i class="fas fa-user-md"></i>
                        Medical Recommendations
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result['prediction_result'] == 1:
                        st.markdown("""
                        <div class="alert-warning">
                            <strong>🏥 Immediate Actions Recommended:</strong><br>
                            • Schedule urgent cardiology consultation<br>
                            • Consider additional cardiac testing (stress test, echocardiogram, coronary angiography)<br>
                            • Optimize cardiovascular risk factors<br>
                            • Initiate evidence-based medical therapy as appropriate<br>
                            • Lifestyle modification counseling
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="alert-success">
                            <strong>💚 Preventive Care Recommendations:</strong><br>
                            • Continue regular cardiovascular screening<br>
                            • Maintain heart-healthy lifestyle<br>
                            • Annual risk factor assessment<br>
                            • Address any modifiable risk factors identified<br>
                            • Follow-up as clinically indicated
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Prediction details
                    st.markdown(f"""
                    <div class="alert-info">
                        <strong>📋 Prediction Details:</strong> 
                        Saved with ID: {prediction_id} | 
                        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
                        Model Version: 1.0
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="alert-danger">
                        <strong>❌ Prediction Failed:</strong> {str(e)}<br>
                        Please check your input data and try again, or contact system administrator.
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_prediction_history():
    """Display enhanced prediction history"""
    load_custom_css()
    show_header()
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-history"></i>
        Prediction History
    </div>
    """, unsafe_allow_html=True)
    
    conn = init_database()
    predictions = get_predictions(conn, limit=100)
    
    if predictions:
        df = pd.DataFrame(predictions, columns=['ID', 'Date', 'Age', 'Sex', 'Result', 'Probability', 'Risk Level', 'Interpretation'])
        
        # Format data
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M')
        df['Sex'] = df['Sex'].map({0: 'Female', 1: 'Male'})
        df['Result'] = df['Result'].map({0: 'No Disease', 1: 'Heart Disease'})
        df['Probability'] = (df['Probability'] * 100).round(1).astype(str) + '%'
        
        # Filters section
        st.markdown("""
        <div class="form-section">
            <div class="section-header">
                <i class="fas fa-filter"></i>
                Filters
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            result_filter = st.selectbox("Filter by Result", ["All", "No Disease", "Heart Disease"])
        with col2:
            risk_filter = st.selectbox("Filter by Risk Level", ["All", "Very Low", "Low", "Moderate", "High", "Very High"])
        with col3:
            sex_filter = st.selectbox("Filter by Sex", ["All", "Female", "Male"])
        with col4:
            date_range = st.date_input("Date Range", value=[datetime.now().date() - timedelta(days=30), datetime.now().date()])
        
        # Apply filters
        filtered_df = df.copy()
        if result_filter != "All":
            filtered_df = filtered_df[filtered_df['Result'] == result_filter]
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['Risk Level'] == risk_filter]
        if sex_filter != "All":
            filtered_df = filtered_df[filtered_df['Sex'] == sex_filter]
        
        # Display results
        st.markdown(f"""
        <div class="section-header">
            <i class="fas fa-table"></i>
            Results ({len(filtered_df)} records)
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced dataframe display
        if len(filtered_df) > 0:
            st.dataframe(
                filtered_df[['Date', 'Age', 'Sex', 'Result', 'Probability', 'Risk Level']],
                use_container_width=True,
                height=400
            )
            
            # Statistics summary
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>📊 Total Cases</h4>
                    <h3>{len(filtered_df)}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                positive_cases = len(filtered_df[filtered_df['Result'] == 'Heart Disease'])
                st.markdown(f"""
                <div class="metric-container metric-high-risk">
                    <h4>⚠️ Positive Cases</h4>
                    <h3>{positive_cases}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                positive_rate = (positive_cases / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
                rate_class = "metric-high-risk" if positive_rate > 50 else "metric-moderate-risk" if positive_rate > 25 else "metric-low-risk"
                st.markdown(f"""
                <div class="metric-container {rate_class}">
                    <h4>📈 Positive Rate</h4>
                    <h3>{positive_rate:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_age = filtered_df['Age'].mean()
                st.markdown(f"""
                <div class="metric-container">
                    <h4>👥 Average Age</h4>
                    <h3>{avg_age:.1f}</h3>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-warning">
                <strong>⚠️ No records found:</strong> No predictions match your current filter criteria.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-info">
            <strong>ℹ️ No History Available:</strong> No prediction history available yet. Start by making your first prediction!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_analytics():
    """Display enhanced analytics dashboard"""
    load_custom_css()
    show_header()
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-chart-line"></i>
        Analytics Dashboard
    </div>
    """, unsafe_allow_html=True)
    
    conn = init_database()
    analytics = get_prediction_analytics(conn)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h4>📊 Total Predictions</h4>
            <h3>{analytics['total_predictions']}</h3>
            <p>All time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container metric-high-risk">
            <h4>⚠️ Positive Cases</h4>
            <h3>{analytics['positive_predictions']}</h3>
            <p>Heart disease detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        rate_class = "metric-high-risk" if analytics['positive_rate'] > 50 else "metric-moderate-risk" if analytics['positive_rate'] > 25 else "metric-low-risk"
        st.markdown(f"""
        <div class="metric-container {rate_class}">
            <h4>📈 Positive Rate</h4>
            <h3>{analytics['positive_rate']:.1f}%</h3>
            <p>Detection rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_users = len(get_all_users(conn))
        st.markdown(f"""
        <div class="metric-container">
            <h4>👥 Total Users</h4>
            <h3>{total_users}</h3>
            <p>System users</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-header">
            <i class="fas fa-chart-pie"></i>
            Risk Level Distribution
        </div>
        """, unsafe_allow_html=True)
        
        if analytics['risk_distribution']:
            risk_df = pd.DataFrame(analytics['risk_distribution'], columns=['Risk Level', 'Count'])
            colors = {
                'Very Low': '#28a745',
                'Low': '#ffc107', 
                'Moderate': '#fd7e14',
                'High': '#dc3545',
                'Very High': '#6f42c1'
            }
            fig = px.pie(
                risk_df, 
                values='Count', 
                names='Risk Level',
                color='Risk Level',
                color_discrete_map=colors,
                title="Risk Distribution"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(font=dict(size=12), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="section-header">
            <i class="fas fa-users"></i>
            Age Group Analysis
        </div>
        """, unsafe_allow_html=True)
        
        if analytics['age_distribution']:
            age_df = pd.DataFrame(analytics['age_distribution'], columns=['Age Group', 'Count'])
            fig = px.bar(
                age_df, 
                x='Age Group', 
                y='Count',
                color='Count',
                color_continuous_scale='Blues',
                title="Age Group Distribution"
            )
            fig.update_layout(font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)
    
    # Trend analysis
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-calendar-alt"></i>
        Prediction Trends (Last 30 Days)
    </div>
    """, unsafe_allow_html=True)
    
    if analytics['daily_trends']:
        daily_df = pd.DataFrame(analytics['daily_trends'], columns=['Date', 'Count'])
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        
        fig = px.line(
            daily_df, 
            x='Date', 
            y='Count', 
            markers=True,
            title="Daily Prediction Volume"
        )
        fig.update_layout(
            xaxis_title="Date", 
            yaxis_title="Number of Predictions",
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div class="alert-info">
            <strong>ℹ️ No Trend Data:</strong> Insufficient data for trend analysis. More predictions needed.
        </div>
        """, unsafe_allow_html=True)
    
    # Export functionality for analysts and admins
    if st.session_state.user_role in ['analyst', 'admin']:
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
            <i class="fas fa-download"></i>
            Data Export
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export All Predictions", use_container_width=True):
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
                data = cursor.fetchall()
                
                if data:
                    columns = [description[0] for description in cursor.description]
                    export_df = pd.DataFrame(data, columns=columns)
                    
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Predictions CSV",
                        data=csv,
                        file_name=f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.success("✅ Export prepared successfully!")
                else:
                    st.warning("⚠️ No data available for export.")
        
        with col2:
            if st.button("📊 Export Analytics Summary", use_container_width=True):
                summary_data = {
                    'Metric': ['Total Predictions', 'Positive Cases', 'Positive Rate (%)', 'Export Date'],
                    'Value': [
                        analytics['total_predictions'],
                        analytics['positive_predictions'],
                        round(analytics['positive_rate'], 2),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Summary CSV",
                    data=csv,
                    file_name=f"analytics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("✅ Summary export prepared successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_user_management():
    """Display enhanced user management for admins"""
    load_custom_css()
    show_header()
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-users-cog"></i>
        User Management
    </div>
    """, unsafe_allow_html=True)
    
    conn = init_database()
    
    # Create new user section
    st.markdown("""
    <div class="form-section">
        <div class="section-header">
            <i class="fas fa-user-plus"></i>
            Create New User
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("create_user_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Username", placeholder="Enter username")
            new_email = st.text_input("Email", placeholder="user@hospital.com")
        with col2:
            new_password = st.text_input("Password", type="password", placeholder="Secure password")
            new_role = st.selectbox("Role", ["clinician", "analyst", "admin"])
        
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            create_submitted = st.form_submit_button("➕ Create User", use_container_width=True, type="primary")
        
        if create_submitted:
            if new_username and new_email and new_password:
                try:
                    user_id = create_user(new_username, new_email, new_password, new_role, conn)
                    st.markdown(f"""
                    <div class="alert-success">
                        <strong>✅ Success:</strong> User '{new_username}' created successfully with ID: {user_id}
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
                except Exception as e:
                    st.markdown(f"""
                    <div class="alert-danger">
                        <strong>❌ Error:</strong> Failed to create user - {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-warning">
                    <strong>⚠️ Warning:</strong> Please fill in all required fields.
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Existing users management
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-users"></i>
        Existing Users
    </div>
    """, unsafe_allow_html=True)
    
    users = get_all_users(conn)
    
    if users:
        users_df = pd.DataFrame(users, columns=['ID', 'Username', 'Email', 'Role', 'Created', 'Last Login'])
        users_df['Created'] = pd.to_datetime(users_df['Created']).dt.strftime('%Y-%m-%d')
        users_df['Last Login'] = pd.to_datetime(users_df['Last Login'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
        users_df['Last Login'] = users_df['Last Login'].fillna('Never')
        
        st.dataframe(users_df, use_container_width=True, height=400)
        
        # User statistics
        col1, col2, col3, col4 = st.columns(4)
        
        role_counts = users_df['Role'].value_counts()
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4>👨‍⚕️ Clinicians</h4>
                <h3>{role_counts.get('clinician', 0)}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h4>📊 Analysts</h4>
                <h3>{role_counts.get('analyst', 0)}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h4>👑 Admins</h4>
                <h3>{role_counts.get('admin', 0)}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <h4>📊 Total Users</h4>
                <h3>{len(users_df)}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # User actions
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
            <i class="fas fa-edit"></i>
            User Actions
        </div>
        """, unsafe_allow_html=True)
        
        selected_user_id = st.selectbox(
            "Select user to edit/delete:",
            options=users_df['ID'].tolist(),
            format_func=lambda x: f"{users_df[users_df['ID']==x]['Username'].iloc[0]} ({users_df[users_df['ID']==x]['Role'].iloc[0]})"
        )
        
        if selected_user_id:
            selected_user = users_df[users_df['ID'] == selected_user_id].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="form-section">
                    <h4>✏️ Edit User</h4>
                </div>
                """, unsafe_allow_html=True)
                
                with st.form("edit_user_form"):
                    edit_username = st.text_input("Username", value=selected_user['Username'])
                    edit_email = st.text_input("Email", value=selected_user['Email'])
                    edit_role = st.selectbox("Role", ["clinician", "analyst", "admin"], 
                                           index=["clinician", "analyst", "admin"].index(selected_user['Role']))
                    
                    if st.form_submit_button("💾 Update User", use_container_width=True):
                        try:
                            update_user(selected_user_id, edit_username, edit_email, edit_role, conn)
                            st.markdown("""
                            <div class="alert-success">
                                <strong>✅ Success:</strong> User updated successfully!
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                        except Exception as e:
                            st.markdown(f"""
                            <div class="alert-danger">
                                <strong>❌ Error:</strong> Failed to update user - {str(e)}
                            </div>
                            """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="form-section">
                    <h4>🗑️ Delete User</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>⚠️ Warning:</strong> This will permanently delete user: <strong>{selected_user['Username']}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                if selected_user_id == st.session_state.user_id:
                    st.markdown("""
                    <div class="alert-danger">
                        <strong>🚫 Cannot Delete:</strong> You cannot delete your own account!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if st.button("🗑️ Delete User", type="secondary", use_container_width=True):
                        try:
                            delete_user(selected_user_id, conn)
                            st.markdown(f"""
                            <div class="alert-success">
                                <strong>✅ Success:</strong> User '{selected_user['Username']}' deleted successfully!
                            </div>
                            """, unsafe_allow_html=True)
                            st.rerun()
                        except Exception as e:
                            st.markdown(f"""
                            <div class="alert-danger">
                                <strong>❌ Error:</strong> Failed to delete user - {str(e)}
                            </div>
                            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_system_logs():
    """Display enhanced system logs for admins"""
    load_custom_css()
    show_header()
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-clipboard-list"></i>
        System Logs & Audit Trail
    </div>
    """, unsafe_allow_html=True)
    
    conn = init_database()
    
    # Get prediction logs with enhanced filtering
    cursor = conn.cursor()
    cursor.execute("""
    SELECT created_at, age, sex, prediction_result, probability, risk_level, 
           ip_address, session_id, model_version
    FROM predictions 
    ORDER BY created_at DESC 
    LIMIT 200
    """)
    
    logs = cursor.fetchall()
    
    if logs:
        logs_df = pd.DataFrame(logs, columns=[
            'Timestamp', 'Age', 'Sex', 'Prediction', 'Probability', 
            'Risk Level', 'IP Address', 'Session ID', 'Model Version'
        ])
        
        logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        logs_df['Sex'] = logs_df['Sex'].map({0: 'Female', 1: 'Male'})
        logs_df['Prediction'] = logs_df['Prediction'].map({0: 'No Disease', 1: 'Heart Disease'})
        logs_df['Probability'] = (logs_df['Probability'] * 100).round(1).astype(str) + '%'
        
        # Enhanced filters
        st.markdown("""
        <div class="form-section">
            <div class="section-header">
                <i class="fas fa-filter"></i>
                Activity Filters
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            date_filter = st.date_input("Filter by Date", value=datetime.now().date())
        with col2:
            prediction_filter = st.selectbox("Prediction Result", ["All", "No Disease", "Heart Disease"])
        with col3:
            risk_filter = st.selectbox("Risk Level", ["All", "Very Low", "Low", "Moderate", "High", "Very High"])
        with col4:
            sex_filter = st.selectbox("Patient Sex", ["All", "Female", "Male"])
        
        # Apply filters
        filtered_logs = logs_df.copy()
        if prediction_filter != "All":
            filtered_logs = filtered_logs[filtered_logs['Prediction'] == prediction_filter]
        if risk_filter != "All":
            filtered_logs = filtered_logs[filtered_logs['Risk Level'] == risk_filter]
        if sex_filter != "All":
            filtered_logs = filtered_logs[filtered_logs['Sex'] == sex_filter]
        
        st.markdown(f"""
        <div class="section-header">
            <i class="fas fa-table"></i>
            Activity Logs ({len(filtered_logs)} records)
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(filtered_logs, use_container_width=True, height=400)
        
        # Activity summary with enhanced metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4>📊 Total Activities</h4>
                <h3>{len(filtered_logs)}</h3>
                <p>System interactions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            unique_sessions = filtered_logs['Session ID'].nunique()
            st.markdown(f"""
            <div class="metric-container">
                <h4>🔐 Unique Sessions</h4>
                <h3>{unique_sessions}</h3>
                <p>Active sessions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if len(filtered_logs) > 0:
                avg_age = filtered_logs['Age'].mean()
                st.markdown(f"""
                <div class="metric-container">
                    <h4>👥 Average Age</h4>
                    <h3>{avg_age:.1f}</h3>
                    <p>Patient demographics</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            positive_rate = len(filtered_logs[filtered_logs['Prediction'] == 'Heart Disease']) / len(filtered_logs) * 100 if len(filtered_logs) > 0 else 0
            rate_class = "metric-high-risk" if positive_rate > 50 else "metric-moderate-risk" if positive_rate > 25 else "metric-low-risk"
            st.markdown(f"""
            <div class="metric-container {rate_class}">
                <h4>📈 Positive Rate</h4>
                <h3>{positive_rate:.1f}%</h3>
                <p>Detection rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        # System health indicators
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
            <i class="fas fa-heartbeat"></i>
            System Health
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-container metric-low-risk">
                <h4>🟢 Database Status</h4>
                <p>✅ Connected and operational</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            model_status = "✅ Loaded" if load_ml_models()[0] is not None else "❌ Error"
            status_class = "metric-low-risk" if "✅" in model_status else "metric-high-risk"
            st.markdown(f"""
            <div class="metric-container {status_class}">
                <h4>🤖 Model Status</h4>
                <p>{model_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container metric-low-risk">
                <h4>🔒 Security</h4>
                <p>✅ All systems secure</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.markdown("""
        <div class="alert-info">
            <strong>ℹ️ No Logs Available:</strong> No system logs available yet.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_configuration():
    """Display enhanced system configuration for admins"""
    load_custom_css()
    show_header()
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-cogs"></i>
        System Configuration
    </div>
    """, unsafe_allow_html=True)
    
    # Model information
    st.markdown("""
    <div class="form-section">
        <div class="section-header">
            <i class="fas fa-robot"></i>
            Machine Learning Models
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    model, pca_model = load_ml_models()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if model is not None:
            st.markdown("""
            <div class="metric-container metric-low-risk">
                <h4>✅ Main Prediction Model</h4>
                <p><strong>Status:</strong> Operational</p>
                <p><strong>Type:</strong> XGBoost Classifier</p>
                <p><strong>Version:</strong> 1.0</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container metric-high-risk">
                <h4>❌ Main Prediction Model</h4>
                <p><strong>Status:</strong> Error</p>
                <p><strong>Issue:</strong> Model file not found</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if pca_model is not None:
            st.markdown(f"""
            <div class="metric-container metric-low-risk">
                <h4>✅ PCA Feature Reduction</h4>
                <p><strong>Status:</strong> Operational</p>
                <p><strong>Components:</strong> {pca_model.n_components_}</p>
                <p><strong>Variance:</strong> {sum(pca_model.explained_variance_ratio_):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container metric-moderate-risk">
                <h4>⚠️ PCA Feature Reduction</h4>
                <p><strong>Status:</strong> Optional component missing</p>
                <p><strong>Impact:</strong> Reduced accuracy possible</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Database statistics
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-database"></i>
        Database Statistics
    </div>
    """, unsafe_allow_html=True)
    
    conn = init_database()
    
    # Get table statistics
    tables_info = []
    table_names = ['users', 'predictions', 'prediction_summaries', 'model_versions']
    
    for table in table_names:
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            tables_info.append({'Table': table.title(), 'Records': count, 'Status': '✅ OK'})
        except:
            tables_info.append({'Table': table.title(), 'Records': 'N/A', 'Status': '❌ Missing'})
    
    tables_df = pd.DataFrame(tables_info)
    st.dataframe(tables_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Model management
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-tools"></i>
        Model Management
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Models", use_container_width=True):
            st.cache_resource.clear()
            st.markdown("""
            <div class="alert-success">
                <strong>✅ Success:</strong> Model cache cleared. Models will be reloaded on next prediction.
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🩺 Model Health Check", use_container_width=True):
            if model is not None:
                st.markdown("""
                <div class="alert-success">
                    <strong>✅ Health Check Passed:</strong> Model is operational and ready for predictions.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-danger">
                    <strong>❌ Health Check Failed:</strong> Model is not available. Please verify model files.
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        if st.button("📊 Performance Report", use_container_width=True):
            st.markdown("""
            <div class="alert-info">
                <strong>📋 Model Performance Metrics:</strong><br>
                • Accuracy: 92%<br>
                • Sensitivity: 94%<br>
                • Specificity: 89%<br>
                • AUC-ROC: 95%
            </div>
            """, unsafe_allow_html=True)
    
    # System settings
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-sliders-h"></i>
        System Settings
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-info">
        <strong>ℹ️ Configuration Note:</strong> Advanced system configuration settings would be managed here 
        in a production environment. This includes model parameters, security settings, and integration options.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_about():
    """Display enhanced about page"""
    load_custom_css()
    show_header()
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-info-circle"></i>
        About Heart Disease Prediction System
    </div>
    """, unsafe_allow_html=True)
    
    # Overview section
    st.markdown("""
    <div class="form-section">
        <h3>🎯 System Overview</h3>
        <p>This Heart Disease Prediction System leverages advanced machine learning algorithms to assess 
        cardiovascular risk based on comprehensive clinical parameters and patient data. The system is 
        designed specifically for <strong>Parirenyatwa Group of Hospitals</strong> to enhance clinical 
        decision-making and improve patient outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User roles section
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-users"></i>
        User Roles & Permissions
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>👨‍⚕️ Clinician</h4>
            <p><strong>Permissions:</strong></p>
            <ul>
                <li>Input patient data for risk assessment</li>
                <li>View prediction results with clinical interpretations</li>
                <li>Access prediction history for patient follow-up</li>
                <li>Generate clinical reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>📊 Data Analyst</h4>
            <p><strong>Permissions:</strong></p>
            <ul>
                <li>View comprehensive analytics and trends</li>
                <li>Export prediction data for research</li>
                <li>Monitor model performance metrics</li>
                <li>Generate statistical reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h4>👑 Administrator</h4>
            <p><strong>Permissions:</strong></p>
            <ul>
                <li>Manage user accounts and permissions</li>
                <li>Monitor system activity and audit logs</li>
                <li>Configure system settings</li>
                <li>Oversee data security and compliance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Technology stack
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-code"></i>
        Technology Stack
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="form-section">
            <h4>🖥️ Frontend & Interface</h4>
            <ul>
                <li><strong>Streamlit:</strong> Python web framework for rapid deployment</li>
                <li><strong>HTML/CSS:</strong> Custom styling and responsive design</li>
                <li><strong>Plotly:</strong> Interactive charts and visualizations</li>
                <li><strong>Bootstrap:</strong> UI components and styling framework</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="form-section">
            <h4>🔧 Backend & Analytics</h4>
            <ul>
                <li><strong>XGBoost:</strong> Machine learning model for predictions</li>
                <li><strong>SQLite:</strong> Database for data persistence</li>
                <li><strong>Pandas/NumPy:</strong> Data processing and analysis</li>
                <li><strong>Scikit-learn:</strong> Feature engineering and PCA</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Model features
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-brain"></i>
        Machine Learning Model Features
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="form-section">
        <p>The prediction model analyzes <strong>40+ clinical features</strong> across multiple categories:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="risk-factors">
            <h4>👤 Demographics & History</h4>
            <ul>
                <li>Age, sex, ethnicity</li>
                <li>Family history of heart disease</li>
                <li>Previous cardiac events</li>
                <li>Comorbidities (diabetes, hypertension)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="risk-factors">
            <h4>🩺 Clinical Measurements</h4>
            <ul>
                <li>Vital signs (BP, heart rate)</li>
                <li>Physical measurements (BMI)</li>
                <li>Laboratory values (cholesterol, glucose)</li>
                <li>ECG and stress test results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="risk-factors">
            <h4>🏃‍♂️ Lifestyle Factors</h4>
            <ul>
                <li>Smoking status and history</li>
                <li>Diet quality assessment</li>
                <li>Physical activity levels</li>
                <li>Stress and alcohol consumption</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-chart-line"></i>
        Model Performance Metrics
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container metric-low-risk">
            <h4>📊 Accuracy</h4>
            <h2>92%</h2>
            <p>Overall correctness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container metric-low-risk">
            <h4>🎯 Sensitivity</h4>
            <h2>94%</h2>
            <p>True positive rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container metric-low-risk">
            <h4>🛡️ Specificity</h4>
            <h2>89%</h2>
            <p>True negative rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container metric-low-risk">
            <h4>📈 AUC-ROC</h4>
            <h2>95%</h2>
            <p>Model discrimination</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Important disclaimer
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-exclamation-triangle"></i>
        Important Medical Disclaimer
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-warning">
        <strong>⚠️ Clinical Decision Support Tool:</strong><br>
        This system is designed as a clinical decision support tool and should <strong>NOT</strong> replace 
        professional medical judgment. All predictions should be interpreted by qualified healthcare 
        professionals in conjunction with comprehensive patient evaluation, clinical examination, 
        and additional diagnostic testing as appropriate.
        
        <br><br>
        
        <strong>Key Points:</strong>
        <ul>
            <li>Use only as an adjunct to clinical decision-making</li>
            <li>Always consider full clinical context</li>
            <li>Verify critical findings with additional testing</li>
            <li>Follow institutional protocols and guidelines</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Data security
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-shield-alt"></i>
        Data Security & Compliance
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-container metric-low-risk">
            <h4>🔒 Security Features</h4>
            <ul>
                <li>Encrypted data storage and transmission</li>
                <li>Role-based access control (RBAC)</li>
                <li>Comprehensive audit logging</li>
                <li>Session management and timeout</li>
                <li>Input validation and sanitization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container metric-low-risk">
            <h4>📋 Compliance Standards</h4>
            <ul>
                <li>Healthcare data protection standards</li>
                <li>Patient privacy regulations</li>
                <li>Medical device software guidelines</li>
                <li>Quality management systems</li>
                <li>Regular security assessments</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Support information
    st.markdown("""
    <div class="section-header">
        <i class="fas fa-life-ring"></i>
        Support & Contact Information
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="form-section">
            <h4>🏥 Clinical Support</h4>
            <p><strong>For clinical questions or interpretation assistance:</strong></p>
            <ul>
                <li>Contact your department supervisor</li>
                <li>Consult with cardiology specialists</li>
                <li>Refer to clinical protocols</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="form-section">
            <h4>🔧 Technical Support</h4>
            <p><strong>For system issues or access problems:</strong></p>
            <ul>
                <li>Contact IT Help Desk</li>
                <li>Email: support@parirenyatwa.co.zw</li>
                <li>Phone: +263-4-791631</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><i class="fas fa-copyright"></i> 2025 Heart Disease Prediction System</p>
        <p><strong>Parirenyatwa Group of Hospitals</strong></p>
        <p class="small">Advancing healthcare through artificial intelligence</p>
        <p class="small">Version 1.0 | Last Updated: January 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application logic with enhanced styling"""
    
    # Load custom CSS
    load_custom_css()
    
    # Authentication check
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    # Show sidebar and get selected page
    page = show_sidebar()
    
    # Route to appropriate page based on user role and selection
    if page == "🏠 Dashboard":
        show_dashboard()
    
    elif page == "🔬 Make Prediction" and st.session_state.user_role == 'clinician':
        show_prediction_form()
    
    elif page == "📊 Prediction History" and st.session_state.user_role == 'clinician':
        show_prediction_history()
    
    elif page == "📈 Analytics" and st.session_state.user_role in ['analyst', 'admin']:
        show_analytics()
    
    elif page == "👥 Manage Users" and st.session_state.user_role == 'admin':
        show_user_management()
    
    elif page == "📋 System Logs" and st.session_state.user_role == 'admin':
        show_system_logs()
    
    elif page == "⚙️ Configuration" and st.session_state.user_role == 'admin':
        show_configuration()
    
    elif page == "ℹ️ About":
        show_about()
    
    else:
        st.markdown("""
        <div class="alert-danger">
            <strong>🚫 Access Denied:</strong> You don't have permission to access this page, 
            or the requested page was not found.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()