from app.extensions.api import api


def init_app(app, **kwargs):
    from . import controller

    api.add_namespace(controller.ns)
