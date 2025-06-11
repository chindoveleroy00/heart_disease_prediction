from flask import Flask
from flask_bootstrap import Bootstrap

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key'  # Change this in production
    
    # Initialize Flask extensions
    Bootstrap(app)
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)
    
    return app