import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()

# Define the exact path as requested
raw_data_path = r"C:\Users\lchin\Desktop\Projects\Prototypes\heart_disease_prediction\data\raw"
output_file = os.path.join(raw_data_path, "zimbabwe_heart_disease_data.csv")

# Ensure the raw directory exists
os.makedirs(raw_data_path, exist_ok=True)

# Zimbabwe-specific ethnic groups
ethnic_groups = ['Shona', 'Ndebele', 'Manyika', 'Kalanga', 'Tonga', 'White Zimbabwean', 'Asian Zimbabwean',
                 'Mixed Race']


# Generate synthetic dataset
def generate_heart_disease_data(num_samples=60000):
    data = []

    for _ in range(num_samples):
        # Patient demographics
        age = int(np.random.normal(50, 15))
        age = max(18, min(100, age))  # Ensure age is between 18-100
        sex = random.choice(['Male', 'Female'])
        ethnicity = random.choices(
            ethnic_groups,
            weights=[70, 15, 5, 2, 2, 3, 1, 2],  # Approximate Zimbabwe demographics
            k=1
        )[0]

        # Medical history (with age/ethnicity dependencies)
        family_history = random.choices([0, 1], weights=[70, 30])[0]
        prev_cardiac = 0
        if age > 50 and family_history:
            prev_cardiac = random.choices([0, 1], weights=[80, 20])[0]

        diabetes_status = random.choices(['No', 'Prediabetic', 'Yes'], weights=[70, 20, 10])[0]
        if age > 45 and ethnicity in ['Shona', 'Ndebele']:
            diabetes_status = random.choices(['No', 'Prediabetic', 'Yes'], weights=[60, 25, 15])[0]

        hypertension = random.choices([0, 1], weights=[60, 40])[0]
        if age > 40:
            hypertension = random.choices([0, 1], weights=[50, 50])[0]

        # Clinical measurements
        systolic = int(np.random.normal(120, 15))
        diastolic = int(np.random.normal(80, 10))
        if hypertension:
            systolic = int(np.random.normal(140, 20))
            diastolic = int(np.random.normal(90, 15))
        systolic = max(80, min(200, systolic))
        diastolic = max(60, min(120, diastolic))

        resting_hr = int(np.random.normal(72, 10))
        height = int(np.random.normal(165, 10) if sex == 'Female' else np.random.normal(175, 10))
        weight = int(np.random.normal(70, 15) if sex == 'Female' else np.random.normal(80, 20))
        bmi = round(weight / ((height / 100) ** 2), 1)

        # Cholesterol with some correlation to age, weight, and diabetes
        total_chol = int(np.random.normal(190, 40))
        hdl = int(np.random.normal(50, 15))
        ldl = total_chol - hdl - int(np.random.normal(30, 10))
        if diabetes_status == 'Yes' or bmi > 30:
            total_chol = int(np.random.normal(220, 40))
            hdl = int(np.random.normal(40, 10))
            ldl = total_chol - hdl - int(np.random.normal(30, 10))

        # Blood glucose with diabetes dependency
        if diabetes_status == 'No':
            fasting_glucose = int(np.random.normal(90, 10))
            hba1c = round(np.random.normal(5.0, 0.5), 1)
        elif diabetes_status == 'Prediabetic':
            fasting_glucose = int(np.random.normal(110, 10))
            hba1c = round(np.random.normal(5.8, 0.3), 1)
        else:
            fasting_glucose = int(np.random.normal(140, 30))
            hba1c = round(np.random.normal(7.5, 1.5), 1)

        # Lifestyle factors
        smoking = random.choices(['Never', 'Former', 'Current'], weights=[60, 20, 20])[0]
        alcohol = int(np.random.poisson(3))  # drinks per week
        physical_activity = int(np.random.poisson(3))  # hours per week
        diet_quality = random.choices([1, 2, 3, 4, 5], weights=[10, 20, 40, 20, 10])[0]  # 1-5 scale
        stress_level = random.choices([1, 2, 3, 4, 5], weights=[15, 25, 30, 20, 10])[0]  # 1-5 scale

        # Symptoms and test results
        chest_pain = random.choices([
            'Typical angina',
            'Atypical angina',
            'Non-anginal pain',
            'Asymptomatic'
        ], weights=[15, 30, 30, 25])[0]

        exercise_angina = random.choices([0, 1], weights=[70, 30])[0]
        if chest_pain != 'Asymptomatic':
            exercise_angina = random.choices([0, 1], weights=[60, 40])[0]

        resting_ecg = random.choices([
            'Normal',
            'ST-T wave abnormality',
            'Left ventricular hypertrophy'
        ], weights=[70, 20, 10])[0]

        max_hr = 220 - age  # Theoretical max
        max_hr = int(np.random.normal(max_hr * 0.85, 10))

        st_depression = round(np.random.exponential(0.5), 1)
        if chest_pain != 'Asymptomatic':
            st_depression = round(np.random.exponential(1.0), 1)

        st_slope = random.choices([
            'Upsloping',
            'Flat',
            'Downsloping'
        ], weights=[60, 30, 10])[0]

        num_vessels = random.choices([0, 1, 2, 3], weights=[70, 15, 10, 5])[0]
        thal = random.choices([
            'Normal',
            'Fixed defect',
            'Reversible defect'
        ], weights=[80, 10, 10])[0]

        # Target variable - heart disease diagnosis
        # Calculate probability based on risk factors
        risk_score = 0
        risk_score += age / 10
        risk_score += 10 if sex == 'Male' else 5
        risk_score += 15 if family_history else 0
        risk_score += 10 if diabetes_status == 'Yes' else (5 if diabetes_status == 'Prediabetic' else 0)
        risk_score += 10 if hypertension else 0
        risk_score += (bmi - 25) / 5 if bmi > 25 else 0
        risk_score += (total_chol - 200) / 20 if total_chol > 200 else 0
        risk_score += (ldl - 100) / 20 if ldl > 100 else 0
        risk_score += (hdl - 40) / -10 if hdl < 40 else 0
        risk_score += 10 if smoking == 'Current' else (5 if smoking == 'Former' else 0)
        risk_score += stress_level * 2

        # Convert risk score to probability (0-100)
        prob = 1 / (1 + np.exp(-(risk_score - 50) / 10))
        heart_disease = 1 if random.random() < prob else 0

        # Add some noise to make it realistic
        if random.random() < 0.1:  # 10% chance of misclassification
            heart_disease = 1 - heart_disease

        data.append([
            age, sex, ethnicity, family_history, prev_cardiac, diabetes_status, hypertension,
            systolic, diastolic, resting_hr, height, weight, bmi, total_chol, hdl, ldl,
            fasting_glucose, hba1c, smoking, alcohol, physical_activity, diet_quality, stress_level,
            chest_pain, exercise_angina, resting_ecg, max_hr, st_depression, st_slope, num_vessels, thal,
            heart_disease
        ])

    columns = [
        'age', 'sex', 'ethnicity', 'family_history', 'previous_cardiac_events', 'diabetes_status',
        'hypertension', 'systolic_bp', 'diastolic_bp', 'resting_heart_rate', 'height_cm', 'weight_kg',
        'bmi', 'total_cholesterol', 'hdl', 'ldl', 'fasting_glucose', 'hba1c', 'smoking_status',
        'alcohol_per_week', 'physical_activity_hours', 'diet_quality', 'stress_level', 'chest_pain_type',
        'exercise_induced_angina', 'resting_ecg', 'max_heart_rate', 'st_depression', 'st_slope',
        'num_vessels', 'thalassemia', 'heart_disease'
    ]

    return pd.DataFrame(data, columns=columns)


def main():
    # Generate the dataset
    print("Generating synthetic dataset...")
    start_time = datetime.now()
    df = generate_heart_disease_data(60000)
    print(f"Dataset generated in {datetime.now() - start_time}")

    # Save to CSV in the raw folder
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to '{output_file}'")

    # Show sample
    print("\nSample of the dataset:")
    print(df.head())
    return df


if __name__ == "__main__":
    main()