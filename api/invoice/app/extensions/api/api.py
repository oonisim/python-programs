"""
app/extensions/api/api.py
Overrides Flask-Restplus Api class
Forked from https://github.com/frol/flask-restplus-server-example flask-restplus-patched folder
"""
from flask import current_app, jsonify
from flask_restplus import Api as OriginalApi
from werkzeug import cached_property

from .http_exceptions import HTTPStatus
from .swagger import Swagger
from .namespace import Namespace


class Api(OriginalApi):

    @cached_property
    def __schema__(self):
        '''
        The Swagger specifications/schema for this API

        :returns dict: the schema as a serializable dict
        '''
        if not self._schema:
            try:
                self._schema = Swagger(self).as_dict()
            except Exception:
                # Log the source exception for debugging purpose
                # and return an error message
                msg = 'Unable to render schema'
                current_app.logger.exception(msg)  # This will provide a full traceback
                return {'error': msg}
        return self._schema

    def init_app(self, app, **kwargs):
        # This solves the issue of late resources registration:
        # https://github.com/frol/flask-restplus-server-example/issues/110
        # https://github.com/noirbizarre/flask-restplus/pull/483
        self.app = app

        super(Api, self).init_app(app, **kwargs)
        app.errorhandler(HTTPStatus.UNPROCESSABLE_ENTITY.value)(handle_validation_error)

    def namespace(self, *args, **kwargs):
        # The only purpose of this method is to pass a custom Namespace class
        _namespace = Namespace(*args, **kwargs)
        self.add_namespace(_namespace)
        self.namespaces.append(_namespace)
        return _namespace


# Return validation errors as JSON
def handle_validation_error(err):
    exc = err.data['exc']
    return jsonify({
        'status': HTTPStatus.UNPROCESSABLE_ENTITY.value,
        'message': exc.messages
    }), HTTPStatus.UNPROCESSABLE_ENTITY.value
