"""
app/modules/__init__.py
Initialize app modules
"""
from app.extensions.constants import *


def init_app(app, **kwargs):
    from importlib import import_module

    for module_name in app.config.get(ENABLED_MODULES):
        import_module('.%s' % module_name, package=__name__).init_app(app, **kwargs)
