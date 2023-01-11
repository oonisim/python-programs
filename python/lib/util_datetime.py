"""
Python date/time utility
"""
import sys
import logging
import datetime
import calendar
from typing import (
    List,
    Dict,
    Tuple,
    Any,
    Optional,
    Callable
)

import dateutil
from dateutil import relativedelta
from dateutil.tz import tzutc

import datefinder


logger: logging.Logger = logging.getLogger(__name__)


def get_datetime_components(date_time: datetime) -> dict:
    """Extract date/time components as dictionary
    Args:
        date_time: datetime instane to extract components
    """
    return {
        "year": date_time.year,
        "month": date_time.month,
        "day": date_time.day,
        "weekday": date_time.weekday(),
        "hour": date_time.hour,
        "minute": date_time.minute,
        "second": date_time.second,
        "microsecond": date_time.microsecond,
        "tzinfo": date_time.tzinfo,
    }


def parse_date_string_as_string(date_str: str, formats=None, year_first=False) -> datetime.datetime:
    day_first_date_formats = (
        # --------------------------------------------------------------------------------
        # 21/01/01 as 21MAY2021
        # --------------------------------------------------------------------------------
        '%d/%m/%y',
        '%d-%m-%y',
        '%d %m %y',
        # --------------------------------------------------------------------------------
        # 21/01/2001 as 21JAN2021
        # --------------------------------------------------------------------------------
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%d %m %Y',
        # --------------------------------------------------------------------------------
        # 21/Jan/01 as 21JAN2021
        # --------------------------------------------------------------------------------
        '%d/%b/%y',
        '%d-%b-%y',
        '%d %b %y',
        # --------------------------------------------------------------------------------
        # 21/January/01 as 21JAN2021
        # --------------------------------------------------------------------------------
        '%d/%B/%Y',
        '%d-%B-%Y',
        '%d %B %Y',
    )
    year_first_date_formats = (
        # --------------------------------------------------------------------------------
        # 21/01/01 as 21MAY2021
        # --------------------------------------------------------------------------------
        '%y/%m/%d',
        '%y-%m-%d',
        '%y %m %d',
        # --------------------------------------------------------------------------------
        # 21/01/2001 as 21JAN2021
        # --------------------------------------------------------------------------------
        '%Y/%m/%d',
        '%Y-%m-%d',
        '%Y %m %d',
        # --------------------------------------------------------------------------------
        # 21/Jan/01 as 21JAN2021
        # --------------------------------------------------------------------------------
        '%y/%b/%d',
        '%y-%b-%d',
        '%y %b %d',
        # --------------------------------------------------------------------------------
        # 21/January/01 as 21JAN2021
        # --------------------------------------------------------------------------------
        '%Y/%B/%d',
        '%Y-%B-%d',
        '%Y %B %d',
    )
    date_formats = []
    if formats is None:
        # --------------------------------------------------------------------------------
        # Reverse the format string in day_first_date_formats when year first.
        # --------------------------------------------------------------------------------
        if year_first:
            date_formats = year_first_date_formats
        else:
            date_formats = day_first_date_formats

        logger.debug(date_formats)

    else:
        # --------------------------------------------------------------------------------
        # Do not touch given formats if it is not None
        # --------------------------------------------------------------------------------
        date_formats = formats

    parsed: List[datetime.datetime] = []
    for fmt in date_formats:
        try:
            parsed.append(datetime.datetime.strptime(date_str, fmt))
        except ValueError as e:
            pass

    logger.debug(parsed)
    if len(parsed) == 0 or len(parsed) > 1:
        raise RuntimeError(f"invalid date_str [{date_str}]")

    return parsed[0]


def get_dates_from_string(
        text: str,
        strict: bool = True,
) -> List[datetime.datetime]:
    """Extract datet from string
    Args:
        text: string to find a date
        strict: require complete date. July 2016 of Monday will not return datetimes.
    Returns: datetime
    Raises: RuntimeError is no date
    """
    dates = list(datefinder.find_dates(text, source=False, index=False, strict=strict, base_date=None))
    if len(dates) <= 0:
        raise RuntimeError(f"invalid text with no date: [{text}]")

    return dates


def parse_datetime_string(
        date_time_str: str,
        dayfirst: bool = False,
        yearfirst: bool = False,
) -> datetime.datetime:
    """
    Args:
        date_time_str: date/time string to parse
        dayfirst: regard the first digits as day e.g 01/05/09 as 01/May/2009
        yearfirst: regard the first digits as year e.g. 01/05/09 as 2001/May/09
    Returns: python datetime instance
    Raises:
        dateutil.parser.ParserError: Failure to parse a datetime string.
        dateutil.parser.UnknownTimezoneWarning:  timezone cannot be parsed into a tzinfo.
    """
    try:
        return dateutil.parser.parse(date_time_str, dayfirst=dayfirst, yearfirst=yearfirst)
    except dateutil.parser.ParserError as e:
        raise RuntimeError(f"parse_datetime_string() invalid date/time string [{date_time_str}]") from e
    except dateutil.parser.UnknownTimezoneWarning as e:
        raise RuntimeError(f"parse_datetime_string() invalid timezone in [{date_time_str}]") from e


def get_epoch_from_string(literal, format='%d %b %Y %H:%M:%S %Z%z') -> int:
    """Convert literal date/time string into epoch time in seconds
    e.g. "01 Mar 2018 11:00:00 GMT+1000" into 1519866000
    Args:
        literal: string to parse date/time
        format: date/time format of the string
    Returns: epoch (in seconds) as int
    """
    literal = ' '.join(literal.split())
    dt = datetime.datetime.strptime(literal, format)
    return int(dt.timestamp())  # sec


def get_epoch_from_datetime(date_time: datetime.datetime) -> int:
    """Convert literal datetime into epoch time in seconds
    Args:
        date_time: string to parse date/time
    Returns: epoch (in seconds) as int
    """
    return calendar.timegm(date_time.timetuple())


def get_seconds_between_datetimes(
        from_datetime: datetime.datetime,
        to_datetime: datetime.datetime
) -> int:
    """Seconds between two datetime instances
    Args:
        from_datetime: from
        to_datetime: to
    Returns: seconds as int
    """
    start = calendar.timegm(to_datetime.timetuple())
    end = calendar.timegm(from_datetime.timetuple())

    return int((start - end) / (3600 * 24))


def get_datetime_after_duration(
        start_datetime: datetime.datetime,
        years: int = 0,
        months: int = 0,
        weeks: int = 0,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        microseconds: int = 0
) -> datetime:
    """Date/time after duration from the start point
    Args:
        start_datetime: start point datetime
    Returns: datetime instance
    """
    end_datetime = start_datetime + dateutil.relativedelta.relativedelta(
        years=years,
        months=months,
        weeks=weeks,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        microseconds=microseconds
    )
    return end_datetime


def get_elapsed_time(func, arg):
    """
    Args:
        func: function to execute and time it
        arg: arguments to the function
    Returns: datetime.datetime.timedelta object
    """
    start = datetime.datetime.now()
    func()
    end = datetime.datetime.now()
    elapsed = end - start
    return elapsed
