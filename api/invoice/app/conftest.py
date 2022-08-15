import pytest
from flask import testing
from app import create_app
from app.extensions import db


@pytest.fixture(scope='session')
def test_db():
    db.create_all()
    yield db
    db.drop_all()


@pytest.fixture(scope='session')
def flask_app_client():
    flask_app = create_app(flask_config='testing')

    # Flask provides a way to test your application by exposing the Werkzeug test Client
    # and handling the context locals for you.
    testing_client = flask_app.test_client()

    # Establish an application context before running the tests.
    ctx = flask_app.app_context()
    ctx.push()

    yield testing_client  # this is where the testing happens!

    ctx.pop()


@pytest.fixture(scope='session')
def temp_db_instance_helper(request, test_db):
    def temp_db_instance_manager(instance):
        test_db.session.add(instance)
        test_db.session.commit()
        yield instance
        try:
            test_db.session.delete(instance)
            test_db.session.commit()
        except Exception as e:
            pass

    return temp_db_instance_manager
