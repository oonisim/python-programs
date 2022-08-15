"""
app/extensions.flask_sqlalchemy/__init__.py
Initialize overriden SQLAlchemy class
"""

from flask_sqlalchemy import SQLAlchemy as BaseSQLAlchemy
from app.extensions.constants import SQLALCHEMY_DATABASE_URI


class SQLAlchemy(BaseSQLAlchemy):
    """
    Customized Flask-SQLAlchemy adapter with enabled autocommit.
    """
    def __init__(self, *args, **kwargs):
        if 'session_options' not in kwargs:
            kwargs['session_options'] = {}
        kwargs['session_options']['autocommit'] = False
        kwargs['session_options']['autoflush'] = False
        super(SQLAlchemy, self).__init__(*args, **kwargs)

    def init_app(self, app):
        assert SQLALCHEMY_DATABASE_URI in app.config and app.config.get(SQLALCHEMY_DATABASE_URI),\
            "SQLALCHEMY_DATABASE_URI must be configured!"
        super(SQLAlchemy, self).init_app(app)


        @app.after_request
        def after_request(response):
            """
            Execute commit on db after request if the session has object
            that needs to be commited.
            """
            from app.extensions import db
            from sqlalchemy.exc import SQLAlchemyError, IntegrityError
            from app.extensions.api import abort, HTTPStatus

            if response.status_code >= 400:
                return response
            if db.session.new or db.session.dirty or db.session.deleted:
                try:
                    db.session.commit()
                except ValueError as e:
                    db.session.rollback()
                    abort(
                        code=HTTPStatus.CONFLICT
                    )
                except IntegrityError as e:
                    db.session.rollback()
                    abort(
                        code=HTTPStatus.CONFLICT
                    )
                except SQLAlchemyError as e:
                    db.session.rollback()
                    abort(
                        code=HTTPStatus.INTERNAL_SERVER_ERROR
                    )
                except Exception as e:
                    db.session.rollback()
                    abort(
                        code=HTTPStatus.INTERNAL_SERVER_ERROR
                    )
            # db.session.close()
            return response
