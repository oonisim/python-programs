"""Module for Flaks operations
"""
import json
import logging
from typing import (
    List,
    Dict,
    Any,
)
import flask


def get_payload_from_json(
        data: Dict[str, Any],
        names: List[str]
) -> Dict[str, Any]:
    """
    Args:
        data: JSON
        names: list of element names to extract
    Returns: Dict
    Raises:
        ValueError: Invalid payload element data.
    """
    assert isinstance(data, dict), f"expected JSON/Dict got {type(data)}."
    return {
        _name: data.get(_name, None)
        for _name in names
    }

