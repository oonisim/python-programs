import logging
import requests
import pandas as pd


logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger(__name__)


def split(tasks: pd.DataFrame, num: int):
    """Split tasks into num assignments and dispense them sequentially
    Args:
        tasks: tasks to split into assignments
        num: number of assignments to create
    Yields: An assignment, which is a slice of the tasks
    """
    assert num > 0
    assert len(tasks) > 0
    Logger.debug(f"createing {num} assignments for {len(tasks)} tasks")

    # Total size of the tasks
    total = len(tasks)

    # Each assignment has 'quota' size which can be zero if total < number of assignments.
    quota = int(total / num)

    # Left over after each assignment takes its 'quota'
    redisual = total % num

    start = 0
    while start < total:
        # As long as redisual is there, each assginemt has (quota + 1) as its tasks.
        if redisual > 0:
            size = quota + 1
            redisual -= 1
        else:
            size = quota

        end = start + size
        yield tasks[start : min(end, total)]

        start = end
        end += size


def http_get_content(url, headers):
    """HTTP GET URL content
    Args:
        url: URL to GET
    Returns:
        Content of the HTTP GET response body, or None
    Raises:
        ConnectionError if HTTP status is not 200
    """
    Logger.debug("http_get_content(): GET url [%s] headers [%s]" % (url, headers))

    error = None
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        if response.status_code == 200:
            content = response.content.decode("utf-8")
            return content
    except requests.exceptions.HTTPError as e:
        error = e
        Logger.error("http_get_content(): HTTP error %s" % e)
    except requests.exceptions.ConnectionError as e:
        error = e
        Logger.error("http_get_content(): HTTP error %s" % e)
    except requests.exceptions.Timeout as e:
        error = e
        Logger.error("http_get_content(): HTTP error %s" % e)
    except requests.exceptions.RequestException as e:
        error = e
        Logger.error("http_get_content(): HTTP error %s" % e)

    assert error is not None
    raise RuntimeError("HTTP to SEC EDGAR failed") from error


