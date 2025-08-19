from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_moment import Moment

# Initialize extensions
db = SQLAlchemy()
moment = Moment()


def create_app(config_name='default'):
    app = Flask(__name__)

    # Import configuration
    from app.config import config
    app.config.from_object(config[config_name])

    # Initialize extensions with app
    db.init_app(app)
    moment.init_app(app)

    # Register blueprints
    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    # Import models after db initialization to avoid circular imports
    from app import models

    # Create tables
    with app.app_context():
        db.create_all()

    return app