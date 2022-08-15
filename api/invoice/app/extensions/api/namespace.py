"""
app/extensions/api/namespace.py
Overrides Flask-Restplus Namespace class with useful decorators
Forked from https://github.com/frol/flask-restplus-server-example flask-restplus-patched folder
"""
from contextlib import contextmanager
import sqlalchemy

from flask_restplus._http import HTTPStatus

from functools import wraps

import flask
import flask_marshmallow
from flask_restplus import Namespace as OriginalNamespace
from flask_restplus.utils import merge, unpack
from webargs.flaskparser import parser as webargs_parser
from werkzeug import cached_property, exceptions as http_exceptions

from .webargs_parser import CustomWebargsParser
from .model import Model, DefaultHTTPErrorSchema

class Namespace(OriginalNamespace):
    """
    Having app-specific handlers here.
    """

    WEBARGS_PARSER = CustomWebargsParser()

    def resolve_object_by_model(self, model, object_arg_name, identity_arg_names=None):
        """
        A helper decorator to resolve DB record instance by id.

        Arguments:
            model (type) - a Flask-SQLAlchemy model class with
                ``query.get_or_404`` method
            object_arg_name (str) - argument name for a resolved object
            identity_arg_names (tuple) - a list of argument names holding an
                object identity, by default it will be auto-generated as
                ``%(object_arg_name)s_id``.

        Example:
        >>> @namespace.resolve_object_by_model(User, 'user')
        >>> @namespace.resolve_object_by_model(User, 'user')
        ... def get_user_by_id(user):
        ...     return user
        >>> get_user_by_id(user_id=3)
        <User(id=3, ...)>

        >>> @namespace.resolve_object_by_model(MyModel, 'my_model', ('user_id', 'model_name'))
        ... def get_object_by_two_primary_keys(my_model):
        ...     return my_model
        >>> get_object_by_two_primary_keys(user_id=3, model_name="test")
        <MyModel(user_id=3, name="test", ...)>
        """
        if identity_arg_names is None:
            identity_arg_names = ('%s_id' % object_arg_name, )
        elif not isinstance(identity_arg_names, (list, tuple)):
            identity_arg_names = (identity_arg_names, )
        return self.resolve_object(
            object_arg_name,
            resolver=lambda kwargs: model.query.get_or_404(
                [kwargs.pop(identity_arg_name) for identity_arg_name in identity_arg_names]
            )
        )

    def model(self, name=None, model=None, mask=None, **kwargs):
        """
        A decorator which registers a model (aka schema / definition).

        This extended implementation auto-generates a name for
        ``Flask-Marshmallow.Schema``-based instances by using a class name
        with stripped off `Schema` prefix.
        """
        if isinstance(model, (flask_marshmallow.Schema, flask_marshmallow.base_fields.FieldABC)):
            if not name:
                name = model.__class__.__name__
            if name.endswith('Schema'):
                name = name[:-len('Schema')]
            api_model = Model(name, model, mask=mask)
            api_model.__apidoc__ = kwargs
            return self.add_model(name, api_model)
        return super(Namespace, self).model(name=name, model=model, **kwargs)

    def _register_access_restriction_decorator(self, func, decorator_to_register):
        # pylint: disable=invalid-name
        """
        Helper function to register decorator to function to perform checks
        in options method
        """
        if not hasattr(func, '_access_restriction_decorators'):
            func._access_restriction_decorators = []  # pylint: disable=protected-access
        func._access_restriction_decorators.append(decorator_to_register)  # pylint: disable=protected-access

    def _handle_api_doc(self, cls, doc):
        if doc is False:
            cls.__apidoc__ = False
            return
        cls.__apidoc__ = merge(getattr(cls, '__apidoc__', {}), doc)

    def resolve_object(self, object_arg_name, resolver):
        """
        A helper decorator to resolve object instance from arguments (e.g. identity).

        Example:
        >>> @namespace.route('/<int:user_id>')
        ... class MyResource(Resource):
        ...    @namespace.resolve_object(
        ...        object_arg_name='user',
        ...        resolver=lambda kwargs: User.query.get_or_404(kwargs.pop('user_id'))
        ...    )
        ...    def get(self, user):
        ...        # user is a User instance here
        """
        def decorator(func_or_class):
            if isinstance(func_or_class, type):
                # Handle Resource classes decoration
                # pylint: disable=protected-access
                func_or_class._apply_decorator_to_methods(decorator)
                return func_or_class

            @wraps(func_or_class)
            def wrapper(*args, **kwargs):
                kwargs[object_arg_name] = resolver(kwargs)
                return func_or_class(*args, **kwargs)
            return wrapper
        return decorator

    def parameters(self, parameters, locations=None):
        """
        Endpoint parameters registration decorator.
        """
        def decorator(func):
            if locations is None and parameters.many:
                _locations = ('json', )
            else:
                _locations = locations
            if _locations is not None:
                parameters.context['in'] = _locations

            return self.doc(params=parameters)(
                self.response(code=HTTPStatus.UNPROCESSABLE_ENTITY)(
                    self.WEBARGS_PARSER.use_args(parameters, locations=_locations)(
                        func
                    )
                )
            )
        return decorator

    def response(self, model=None, code=HTTPStatus.OK, description=None, **kwargs):
        """
        Endpoint response OpenAPI documentation decorator.

        It automatically documents HTTPError%(code)d responses with relevant
        schemas.

        Arguments:
            model (flask_marshmallow.Schema) - it can be a class or an instance
                of the class, which will be used for OpenAPI documentation
                purposes. It can be omitted if ``code`` argument is set to an
                error HTTP status code.
            code (int) - HTTP status code which is documented.
            description (str)

        Example:
        >>> @namespace.response(BaseTeamSchema(many=True))
        ... @namespace.response(code=HTTPStatus.FORBIDDEN)
        ... def get_teams():
        ...     if not user.is_admin:
        ...         abort(HTTPStatus.FORBIDDEN)
        ...     return Team.query.all()
        """
        code = HTTPStatus(code)
        if code is HTTPStatus.NO_CONTENT:
            assert model is None
        if model is None and code not in {HTTPStatus.ACCEPTED, HTTPStatus.NO_CONTENT}:
            if code.value not in http_exceptions.default_exceptions:
                raise ValueError("`model` parameter is required for code %d" % code)
            model = self.model(
                name='HTTPError%d' % code,
                model=DefaultHTTPErrorSchema(http_code=code)
            )
        if description is None:
            description = code.description

        def response_serializer_decorator(func):
            """
            This decorator handles responses to serialize the returned value
            with a given model.
            """
            def dump_wrapper(*args, **kwargs):
                # pylint: disable=missing-docstring
                response = func(*args, **kwargs)
                extra_headers = None

                if response is None:
                    if model is not None:
                        raise ValueError("Response cannot not be None with HTTP status %d" % code)
                    return flask.Response(status=code)
                elif isinstance(response, flask.Response) or model is None:
                    return response
                elif isinstance(response, tuple):
                    response, _code, extra_headers = unpack(response)
                else:
                    _code = code

                if HTTPStatus(_code) is code:
                    response = model.dump(response).data
                return response, _code, extra_headers

            return dump_wrapper

        def decorator(func_or_class):
            if code.value in http_exceptions.default_exceptions:
                # If the code is handled by raising an exception, it will
                # produce a response later, so we don't need to apply a useless
                # wrapper.
                decorated_func_or_class = func_or_class
            elif isinstance(func_or_class, type):
                # Handle Resource classes decoration
                # pylint: disable=protected-access
                func_or_class._apply_decorator_to_methods(response_serializer_decorator)
                decorated_func_or_class = func_or_class
            else:
                decorated_func_or_class = wraps(func_or_class)(
                    response_serializer_decorator(func_or_class)
                )

            if model is None:
                api_model = None
            else:
                if isinstance(model, Model):
                    api_model = model
                else:
                    api_model = self.model(model=model)
                if getattr(model, 'many', False):
                    api_model = [api_model]

            doc_decorator = self.doc(
                    responses={
                            code.value: (description, api_model)
                        }
                )
            return doc_decorator(decorated_func_or_class)
        return decorator

    def route(self, *args, **kwargs):
        base_wrapper = super(Namespace, self).route(*args, **kwargs)
        def wrapper(cls):
            if 'OPTIONS' in cls.methods:
                cls.options = self.preflight_options_handler(
                    self.response(code=HTTPStatus.NO_CONTENT)(cls.options)
                )
            return base_wrapper(cls)

        return wrapper
