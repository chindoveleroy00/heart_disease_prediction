# Heart Disease Prediction System

## Overview
The Heart Disease Prediction System is a machine learning-based application designed to assist healthcare providers in identifying patients at risk for heart disease early on. Developed specifically with Zimbabwe's healthcare challenges in mind, this system leverages patient data from Parirenyatwa Group of Hospitals to provide actionable insights for early interventions.

## Key Features
- **Predictive Analytics**: Utilizes the XGBoost machine learning algorithm to assess heart disease risk
- **Early Detection**: Identifies at-risk individuals before symptoms manifest
- **Comprehensive Risk Assessment**: Evaluates multiple factors including:
  - Demographic information
  - Medical history
  - Vital signs
  - Lab results
  - Lifestyle factors
- **Risk Stratification**: Classifies patients into risk categories (Very Low to Very High)
- **Web Interface**: User-friendly form for healthcare providers to input patient data
- **API Endpoint**: Supports integration with existing hospital systems

## Technical Specifications
### Backend
- **Machine Learning Framework**: Scikit-learn, XGBoost
- **Feature Engineering**:
  - Clinical feature creation (age decile, cholesterol ratio, BMI categories)
  - Interaction terms (age-cholesterol, BMI-heart rate)
  - Composite risk scores
  - PCA for dimensionality reduction
- **Model Serving**: Flask-based web application

### Frontend
- **Web Framework**: Flask with Bootstrap
- **Form Validation**: WTForms with comprehensive input validation
- **Result Visualization**: Clear presentation of prediction results with risk interpretation

### Data Requirements
The system requires the following patient data for accurate predictions:
- **Demographics**: Age, sex, ethnicity
- **Medical History**: Family history, diabetes status, hypertension, previous cardiac events
- **Vital Signs**: Blood pressure, heart rate, height, weight
- **Lab Results**: Cholesterol levels, glucose, HbA1c
- **Lifestyle Factors**: Smoking status, alcohol consumption, physical activity
- **Clinical Tests**: ECG results, exercise-induced angina, ST depression

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the application (set SECRET_KEY and other environment variables)
4. Run the Flask application:
   ```bash
   flask run
   ```

## Usage
1. Access the web interface at `http://localhost:5000`
2. Fill in the patient data form with all required information
3. Submit the form to receive the prediction
4. Review the risk assessment and recommended actions

For API usage:
```bash
POST /api/predict
Content-Type: application/json

{
  "age": 45,
  "sex": 1,
  "systolic_bp": 130,
  ...
}
```

## Project Significance
This system addresses critical gaps in Zimbabwe's healthcare system by:
- Moving from reactive to proactive healthcare through predictive analytics
- Optimising resource allocation by identifying high-risk patients
- Providing locally-tailored predictions using data from Parirenyatwa Group of Hospitals
- Reducing healthcare costs through early intervention opportunities

## Future Enhancements
- Integration with hospital EHR systems
- Mobile application for remote assessments
- Additional predictive models for related conditions (diabetes, hypertension)
- Continuous model retraining with new patient data


## Acknowledgments
Parirenyatwa Group of Hospitals for providing the clinical context and data requirements for this project.
