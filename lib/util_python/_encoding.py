"""Module for encoding utility"""
import base64


def is_base64_encoded(data: bytes):
    """check if data is base64 encoded
    Args:
        data: data to check if base64 encoded
    Returns; True if base64 encoded, otherwise False
    """
    try:
        return base64.b64encode(base64.b64decode(data)) == data
    except (TypeError, ValueError):
        return False
