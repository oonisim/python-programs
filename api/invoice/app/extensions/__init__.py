"""
app/extensions/__init__.py
Initialize extensions
"""

from . import api

from .flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()


def init_app(app):
    """
    Application extensions initialization.
    """
    for extension in (
        api,
        db,
    ):
        extension.init_app(app)
