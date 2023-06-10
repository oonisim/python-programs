"""Module for AWS common utilities"""
import logging
import os
from typing import (
    List,
    Optional
)

from util_logging import (
    get_logger
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------
def get_aws_region():
    """Get region from AWS_DEFAULT_REGION or AWS_REGION environment variable
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
    https://stackoverflow.com/q/59961939/4281353
    https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html

    Raises:
        RuntimeError: AWS_DEFAULT_REGION nor AWS_REGION is set
    """
    _name: str = "get_aws_region()"
    region: Optional[str] = None
    if 'AWS_DEFAULT_REGION' in os.environ:
        region = os.getenv('AWS_DEFAULT_REGION')
    elif 'AWS_REGION' in os.environ:
        region = os.getenv('AWS_REGION')
    else:
        msg: str = "AWS_DEFAULT_REGION nor AWS_REGION defined in the environment."
        _logger.error("%s: %s", _name, msg)
        raise RuntimeError(msg)
