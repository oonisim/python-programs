"""Module for common AWS handling class
"""
# pylint: disable=too-few-public-methods
import re
import logging


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# AWS Base class
# --------------------------------------------------------------------------------
class Base:
    """Class to provide base functionalities."""
    @staticmethod
    def validate_text(text: str):
        """Validate if the text is string with length
        Args:
            text: text to validate
        Raises:
            ValueError: text is invalid
        """
        name: str = "validate_text()"
        if not text or not isinstance(text, str):
            msg: str = f"expected the text as a valid string, got [{text}]."
            _logger.error("%s: %s", name, msg)
            raise ValueError(msg)

        text = re.sub(r'[\s\'\"\n\t\\]+', ' ', text, flags=re.MULTILINE).strip()
        if len(text) == 0:
            msg: str = f"expected the text as a valid string with length, got [{text}]."
            _logger.error("%s: %s", name, msg)
            raise ValueError(msg)

        return text
