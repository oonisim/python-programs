"""
HTTP module using Python requests module.
See https://requests.readthedocs.io/en/latest/api/#module-requests.
See https://docs.python.org/3/library/http.html for HTTP status codes.
"""
import os
import shutil
import logging
from http import (
    HTTPStatus
)
from functools import (
    wraps
)
from typing import (
    Dict,
    Generator,
    Optional
)

import requests
from requests.models import (
    Response
)
from requests.exceptions import (
    HTTPError,
    Timeout,
    # Causes "Redefining built-in 'ConnectionError' (redefined-builtin)" which cannot be disabled
    # See https://github.com/pylint-dev/pylint/issues/3677
    # ConnectionError,
    RequestException
)

from util_logging import (
    get_logger,
)
_logger: logging.Logger = get_logger(name=__name__)


# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
TIMEOUT_DEFAULT: float = 30.0


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
def handle_requests_exception(function):
    """Decorator to handle requests exceptions
    Raises: RuntimeError if request failed
    """
    @wraps(handle_requests_exception)
    def wrapper(*args, **kwds):
        name: str = f"{handle_requests_exception.__name__}()"

        try:
            return function(*args, **kwds)
        except (
                HTTPError, requests.exceptions.ConnectionError, Timeout, RequestException
        ) as error:
            _logger.error("%s: failed due to timeout error [%s]", name, error)
            raise RuntimeError(f"{name} failed.") from error

    return wrapper


@handle_requests_exception
def exists_url(
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = TIMEOUT_DEFAULT
) -> bool:
    """Check if URL exists
    Args:
        url: URL to check
        headers:
            HTTP headers to set to the request, e.g {
                "User-Agent": "Company Name myname@company.com"
            }
        timeout: (optional) How many seconds to wait for the server to send data
    Returns: True if exists or False
    Raises:
        RuntimeError if HTTP status is not 200 and not in status_codes_to_ignore
    """
    response: Response = requests.head(url=url, headers=headers, timeout=timeout)
    if response.status_code not in [HTTPStatus.OK, HTTPStatus.NOT_FOUND]:
        response.raise_for_status()

    return response.status_code == HTTPStatus.OK


@handle_requests_exception
def get_content_from_url(
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = TIMEOUT_DEFAULT
) -> str:
    """Get content from url.
    See https://requests.readthedocs.io/en/latest/
    
    Args:
        url: URL to GET
        headers:
            HTTP headers to set to the request, e.g {
                "User-Agent": "Company Name myname@company.com"
            }
        timeout: (optional) How many seconds to wait for the server to send data
    Returns:
        Content of the HTTP GET response body or None if status code is in status_codes_to_ignore
    Raises:
        RuntimeError if HTTP status is not 200 and not in status_codes_to_ignore
    """
    name: str = "get_content_from_url()"
    _logger.debug("%s: url [%s] headers [%s]", name, url, headers)

    response: Response = requests.get(url=url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.content.decode("utf-8")


@handle_requests_exception
def download_from_url(
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = TIMEOUT_DEFAULT,
        path_to_file: Optional[str] = None
) -> str:
    """Download content from URL to the path_to_file.
    Args:
        url: url to download content from
        headers:
            HTTP headers to set to the request, e.g {
                "User-Agent": "Company Name myname@company.com"
            }
        timeout: (optional) How many seconds to wait for the server to send data
        path_to_file:
            path to the file to save the content.
            If None, the filename from url is created at the current directory.

    Returns: filename where the content is saved
    Raises:
        RuntimeError if HTTP status is not OK

    """
    name: str = "download_from_url()"
    _logger.debug(
        "%s: downloading from url [%s] headers [%s] to [%s]", name, url, headers, path_to_file
    )

    filename: str = url.strip().split(os.sep)[-1] if path_to_file is None else path_to_file
    with requests.get(url=url, headers=headers, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with open(path_to_file, 'wb') as _file:
            shutil.copyfileobj(response.raw, _file)

    return filename


@handle_requests_exception
def stream_from_url(
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = TIMEOUT_DEFAULT,
        chunk_size: int = 512,
        decode_unicode: bool = False,
        delimiter: str = None
) -> Generator:
    """Stream content from URL as a generator
    See:
        https://requests.readthedocs.io/en/latest/user/advanced/#body-content-workflow
        https://requests.readthedocs.io/en/latest/api/#requests.Response.iter_lines

    Args:
        url: url to download content from
        headers:
            HTTP headers to set to the request, e.g {
                "User-Agent": "Company Name myname@company.com"
            }
        timeout: (optional) How many seconds to wait for the server to send data
        chunk_size: the number of bytes to read into memory.
        decode_unicode: if True, content will be decoded using the best available encoding based on the response.
        delimiter: character to split the content into lines

    Returns: generator to stream from the url
    Raises:
        RuntimeError if HTTP status is not OK

    """
    name: str = "stream_from_url()"
    _logger.debug(
        "%s: streaming from url [%s] headers [%s]", name, url, headers
    )

    with requests.get(url=url, headers=headers, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines(
                chunk_size=chunk_size, decode_unicode=decode_unicode, delimiter=delimiter
        ):
            yield line
