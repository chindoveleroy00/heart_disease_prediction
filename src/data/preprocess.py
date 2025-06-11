import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Define paths using the specific directory structure
raw_data_path = r"C:\Users\lchin\Desktop\Projects\Prototypes\heart_disease_prediction\data\raw\zimbabwe_heart_disease_data.csv"
processed_dir = r"C:\Users\lchin\Desktop\Projects\Prototypes\heart_disease_prediction\data\processed"
processed_data_path = os.path.join(processed_dir, 'processed_heart_data.csv')


def load_and_preprocess_data():
    """Load raw data and perform preprocessing"""
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Load raw data
    print(f"Loading data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)

    # 1. Handle missing values (if any)
    print("Handling missing values...")
    df.fillna({
        'resting_heart_rate': df['resting_heart_rate'].median(),
        'hdl': df['hdl'].median(),
        'ldl': df['ldl'].median(),
        'st_depression': 0,  # Assuming no depression if missing
        'num_vessels': 0  # Assuming no vessels if missing
    }, inplace=True)

    # 2. Convert categorical variables
    print("Encoding categorical features...")
    categorical_cols = [
        'sex', 'ethnicity', 'diabetes_status', 'smoking_status',
        'chest_pain_type', 'resting_ecg', 'st_slope', 'thalassemia'
    ]

    # Create a dictionary to store label encoders
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])

    # 3. Feature engineering
    print("Creating new features...")
    df['bp_category'] = pd.cut(df['systolic_bp'],
                               bins=[0, 120, 130, 140, 180, 200],
                               labels=[0, 1, 2, 3, 4])

    df['chol_ratio'] = df['total_cholesterol'] / df['hdl']
    df['bmi_category'] = pd.cut(df['bmi'],
                                bins=[0, 18.5, 25, 30, 100],
                                labels=[0, 1, 2, 3])

    # 4. Handle outliers
    print("Handling outliers...")
    numeric_cols = [
        'age', 'systolic_bp', 'diastolic_bp', 'resting_heart_rate',
        'height_cm', 'weight_kg', 'total_cholesterol', 'hdl', 'ldl',
        'fasting_glucose', 'hba1c', 'max_heart_rate', 'st_depression'
    ]

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # 5. Normalize numerical features
    print("Normalizing numerical features...")
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 6. Split into features and target
    X = df.drop('heart_disease', axis=1)
    y = df['heart_disease']

    # 7. Save processed data
    print("Saving processed data...")
    df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

    return df, X, y


if __name__ == "__main__":
    df, X, y = load_and_preprocess_data()
    print("\nPreprocessing complete!")
    print(f"Final dataset shape: {df.shape}")
    print("\nSample of processed data:")
    print(df.head())