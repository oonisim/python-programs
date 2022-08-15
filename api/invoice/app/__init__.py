"""
app/__init__.py
Flask app factory, loads configuration and initialize extensions and modules.
"""
import os

from . import extensions
from . import modules


CONFIG_NAME_MAPPER = {
    'development': 'config.DevelopmentConfig',
    'testing': 'config.TestingConfig',
    'production': 'config.ProductionConfig'
}


def create_app(flask_config='development', **kwargs):
    """
    Entry point to the Flask RESTful Server application.
    """
    import config

    from flask import Flask
    app = Flask(__name__, **kwargs)

    try:
        app.config.from_object(CONFIG_NAME_MAPPER[os.getenv('FLASK_CONFIG', flask_config)])
    except ImportError:
        raise RuntimeError(
            "Error : FLASK_CONFIG incorrectly set in the environment. Make sure you are correctly setting enviroment file."\
            "If you want to use default development configuration (unsafe for production), unset FLASK_CONFIG."
        )

    extensions.init_app(app)
    modules.init_app(app)

    with app.app_context():
        from app.extensions import db
        db.create_all()

    return app
