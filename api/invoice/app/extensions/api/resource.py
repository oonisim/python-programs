"""
app/extensions/api/resource.py
Overrides FlaskRestplus Resource class
Forked from https://github.com/frol/flask-restplus-server-example flask-restplus-patched folder
"""
import flask
from flask_restplus import Resource as OriginalResource
from werkzeug.exceptions import HTTPException

from .http_exceptions import HTTPStatus

class Resource(OriginalResource):
    """
    Extended Flast-RESTPlus Resource to add options method
    """

    @classmethod
    def _apply_decorator_to_methods(cls, decorator):
        """
        This helper can apply a given decorator to all methods on the current
        Resource.

        NOTE: In contrast to ``Resource.method_decorators``, which has a
        similar use-case, this method applies decorators directly and override
        methods in-place, while the decorators listed in
        ``Resource.method_decorators`` are applied on every request which is
        quite a waste of resources.
        """
        for method in cls.methods:
            method_name = method.lower()
            decorated_method_func = decorator(getattr(cls, method_name))
            setattr(cls, method_name, decorated_method_func)
