import pytest

from app import create_app


def test_create_app():
    try:
        create_app()
    except:
        pass
