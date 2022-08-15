"""
app/modules/api/__init__.py
Initialize flask restplus api
"""
from flask import Blueprint
from app.extensions.api import api


def init_app(app, **kwargs):
    """
    Init of the api Blueprint.
    """
    api_blueprint = Blueprint('api', __name__, url_prefix='/api')
    api.init_app(api_blueprint)
    app.register_blueprint(api_blueprint)
