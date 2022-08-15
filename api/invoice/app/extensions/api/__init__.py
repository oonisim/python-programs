"""
app/extensions/api/__init__.py
Initialize overriden Flask-Restplus Api class
Forked from https://github.com/frol/flask-restplus-server-example flask-restplus-patched folder
"""
from .api import Api
from .namespace import Namespace
from .parameters import (Parameters,
                         SearchParameters,
                         PaginationParameters)
from .resource import Resource
from .model import ModelSchema
from .http_exceptions import abort, HTTPStatus
from app.extensions.constants import *


api = Api(
    version='1.0.0',
    title="InvoicesAPI",
    description="The Xero Invoices API",
    ordered=True,
    doc='/doc/'
)

# Temporary fix concerning https://github.com/noirbizarre/flask-restplus/issues/460
api.namespaces = []


def init_app(app, **kwargs):
    pass
