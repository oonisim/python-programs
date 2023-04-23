"""

"""
import logging
import requests
from typing import (
    List,
    Dict,
)


def get_content_from_url(url: str, headers: Dict[str, str]):
    """HTTP GET URL content using requests module.
    See https://requests.readthedocs.io/en/latest/
    
    Args:
        url: URL to GET
        headers:
            HTTP headers to set to the request, e.g {
                "User-Agent": "Company Name myname@company.com"
            }
    Returns:
        Content of the HTTP GET response body, or None
    Raises:
        RuntimeError if HTTP status is not 200
    """
    logging.debug("http_get_content(): GET url [%s] headers [%s]" % (url, headers))

    error = None
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        if response.status_code == 200:
            content = response.content.decode("utf-8")
            return content
    except requests.exceptions.HTTPError as e:
        error = e
        logging.error("http_get_content(): HTTP error %s" % e)
    except requests.exceptions.ConnectionError as e:
        error = e
        logging.error("http_get_content(): HTTP error %s" % e)
    except requests.exceptions.Timeout as e:
        error = e
        logging.error("http_get_content(): HTTP error %s" % e)
    except requests.exceptions.RequestException as e:
        error = e
        logging.error("http_get_content(): HTTP error %s" % e)

    assert error is not None
    raise RuntimeError("HTTP to SEC EDGAR failed") from error