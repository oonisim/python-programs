import os


class BaseConfig(object):
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    ERROR_404_HELP = False

    ENABLED_MODULES = (
        'invoices',
        'api',
    )

    STATIC_ROOT = os.path.join(PROJECT_ROOT, 'static')

    SWAGGER_UI_JSONEDITOR = True

    DEBUG = False
    TESTING = False


class ProductionConfig(BaseConfig):
    # ORM
    SQLALCHEMY_DATABASE_URI =        os.getenv('SQLALCHEMY_DATABASE_URI')
    SQLALCHEMY_TRACK_MODIFICATIONS = os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS')


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = True

    # ORM
    SQLALCHEMY_DATABASE_URI =        'sqlite:///../database/app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class TestingConfig(BaseConfig):
    DEBUG = True
    TESTING = True

    # ORM
    SQLALCHEMY_DATABASE_URI =        'sqlite:///../database/testing_app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
