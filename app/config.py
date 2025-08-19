import os
from pathlib import Path

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Base configuration class"""
    # Secret key for Flask sessions
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-very-secret-key-change-in-production'

    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              f'sqlite:///{BASE_DIR / "database" / "heart_disease.db"}'

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True

    # Ensure database directory exists
    db_dir = BASE_DIR / "database"
    db_dir.mkdir(exist_ok=True)

    # Model paths
    MODEL_PATH = BASE_DIR / 'data' / 'models' / 'heart_disease_model.joblib'
    PCA_MODEL_PATH = BASE_DIR / 'data' / 'models' / 'pca_model.pkl'

    # Application settings
    DEBUG = False
    TESTING = False

    # Pagination
    PREDICTIONS_PER_PAGE = 10

    # Model version tracking
    MODEL_VERSION = "1.0"


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True  # Log SQL queries in development


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # In production, you might want to use PostgreSQL
    # SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
    #     'postgresql://username:password@localhost/heart_disease_prod'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}