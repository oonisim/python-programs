"""
Python date/time utility
"""
import logging
import datetime
import calendar
from typing import (
    List,
    Dict,
    Tuple,
    Union,
)

import dateutil
# from dateutil.tz import tzutc
import datefinder
import holidays
import numpy as np

from util_constant import (
    TYPE_FLOAT,
)


# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------
WEEKDAY_MON: int = 0
WEEKDAY_TUE: int = 1
WEEKDAY_WED: int = 2
WEEKDAY_THU: int = 3
WEEKDAY_FRI: int = 4
WEEKDAY_SAT: int = 5
WEEKDAY_SUN: int = 6

# --------------------------------------------------------------------------------
# module logger
# --------------------------------------------------------------------------------
logging.basicConfig(level=logging.ERROR)
logger: logging.Logger = logging.getLogger(__name__)


def get_datetime_components(date_time: datetime) -> dict:
    """Extract date/time components as dictionary
    weekday is an integer, where Monday is 0 and Sunday is 6.
    See https://docs.python.org/3/library/datetime.html#datetime.datetime.weekday

    To have Monday as 1 and Sunday as 7, need to use isoweekday()
    See https://docs.python.org/3/library/datetime.html#datetime.datetime.isoweekday

    Args:
        date_time: datetime instance to extract components
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


def convert_date_time_into_datetime(
        date_in_year: datetime.date,
        time_in_day: datetime.time,
        tzinfo: datetime.tzinfo = None,
) -> datetime.datetime:
    """Convert (datetime.date, datetime.time) into datetime instance
    Args:
        date_in_year: datetime.date instance to convert
        time_in_day: time instance to convert
        tzinfo: timezone
    Returns: datetime instance
    """
    assert isinstance(date_in_year, datetime.date), \
        f"expected datetime.date got [{date_in_year}] of type [{type(date_in_year)}"
    assert isinstance(time_in_day, datetime.time), \
        f"expected datetime.time got [{time_in_day}] of type [{type(time_in_day)}"

    return datetime.datetime.combine(date=date_in_year, time=time_in_day, tzinfo=tzinfo)


def convert_date_into_datetime(date_in_year: datetime.date) -> datetime.datetime:
    """Convert datetime.date instance into datetime instance
    Ordinal date is the proleptic Gregorian ordinal, where January 1 of year 1 has ordinal 1.

    Args:
        date_in_year: datetime.date instance to convert
    """
    assert isinstance(date_in_year, datetime.date), \
        f"expected datetime.date got [{date_in_year}] of type [{type(date_in_year)}"

    return datetime.datetime.fromordinal(date_in_year.toordinal())


def convert_time_into_timedelta(time_in_day: datetime.time) -> datetime.timedelta:
    """Convert datetime.time instance in the day to timedelta from midnight
    Convert datetime.time() as a point in a day into a duration since midnight.

    datetime.min = datetime(MINYEAR, 1, 1, tzinfo=None)
    date.min = date(MINYEAR, 1, 1)

    Args:
        time_in_day: time instance to convert
    Return: timedelta
    """
    assert isinstance(time_in_day, datetime.time), \
        f"expected datetime.time got [{time_in_day}] of type [{type(time_in_day)}"
    return datetime.datetime.combine(datetime.date.min, time_in_day) - datetime.datetime.min


def parse_date_string(date_str: str, formats=None, year_first=False) -> datetime.datetime:
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

    logger.debug("$s", parsed)
    if len(parsed) == 0 or len(parsed) > 1:
        raise RuntimeError(f"invalid date_str [{date_str}]")

    return parsed[0]


def get_dates_from_string(
        text: str,
        strict: bool = True,
) -> List[datetime.datetime]:
    """Extract dates from string. Accept text with no date, then return empty list.

    Args:
        text: string to find a date
        strict: require complete date. July 2016 of Monday will not return datetimes.
    Returns: list of datetime instances
    """
    dates = list(datefinder.find_dates(text, source=False, index=False, strict=strict))
    if len(dates) <= 0:
        # raise RuntimeError(f"invalid text with no date: [{text}]")
        logger.warning("get_dates_from_string(): no date in the text [%s]", text)

    return dates


def has_date_in_string(text: str):
    """Check if a date is included in the text
    Args:
        text: string to find a date
    """
    assert isinstance(text, str), f"invalid text [{text}] of type [{type(text)}]."

    dates: List[datetime] = get_dates_from_string(text=text)
    return dates is not None and len(dates) > 0


def parse_datetime_string(
        date_time_str: str,
        default: datetime.datetime = None,
        dayfirst: bool = False,
        yearfirst: bool = False,
) -> datetime.datetime:
    """
    Args:
        date_time_str: date/time string to parse
        default: default datetime which elements specified in date_time_str replace.
        dayfirst: regard the first digits as day e.g 01/05/09 as 01/May/2009
        yearfirst: regard the first digits as year e.g. 01/05/09 as 2001/May/09
    Returns: python datetime instance
    Raises:
        dateutil.parser.ParserError: Failure to parse a datetime string.
        dateutil.parser.UnknownTimezoneWarning:  timezone cannot be parsed into a tzinfo.
    """
    try:
        return dateutil.parser.parse(
            timestr=date_time_str, default=default, dayfirst=dayfirst, yearfirst=yearfirst
        )
    except dateutil.parser.ParserError as e:
        raise RuntimeError(
            f"parse_datetime_string() invalid date/time string [{date_time_str}]"
        ) from e
    except dateutil.parser.UnknownTimezoneWarning as e:
        raise RuntimeError(
            f"parse_datetime_string() invalid timezone in [{date_time_str}]"
        ) from e


def parse_time_string(text: str) -> datetime.time:
    """Parse time string e.g. 01:23:53 or 03:00 or 12 and return datetime.time instance.

    Args:
        text: time string to parse
    Returns: datetime.time instance
    Raises: RuntimeError for invalid time expression
    """
    assert isinstance(text, str), f"invalid text [{text}] of type [{type(text)}]."
    error: Exception = RuntimeError(f"invalid time expression [{text}]. Must be hh:mm:ss or mm:ss or as string")

    numbers: List[str] = "".join(text.split()).split(":")
    if len(numbers) < 1 or len(numbers) > 3:
        raise error

    # The expression must be hh:mm:ss with zero padded e.g. 00:01:09
    # Otherwise 01:10 is interpreted as 01:10:00.
    expression: str
    if len(numbers) == 3:
        expression = f"{numbers[0].zfill(2)}:{numbers[1].zfill(2)}:{numbers[2].zfill(2)}"
    elif len(numbers) == 2:
        expression = f"00:{numbers[0].zfill(2)}:{numbers[1].zfill(2)}"
    elif len(numbers) == 1:
        expression = f"00:00:{numbers[0].zfill(2)}"
    else:
        raise error

    try:
        _time: datetime.time = datetime.time.fromisoformat(expression)
    except ValueError as e:
        raise error from e

    assert isinstance(_time, datetime.time), \
        f"invalid text sting [{text}] generating type [{type(_time)}]"

    return _time


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
    :param microseconds:
    :param seconds:
    :param minutes:
    :param hours:
    :param days:
    :param weeks:
    :param start_datetime:
    :param months:
    :param years:
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


def get_holidays(
        country: str,
        states: List[str] = None,
        years: List[int] = None
) -> Dict[datetime.datetime, str]:
    """Get holidays of the country and state (optional).
    If state is not specified, the common holidays for all states will be provided.

    holidays.utils.country_holidays(
        country, subdiv=None, years=None, expand=True, observed=True
    )
    observed:
        Whether to include the dates of when public holiday are observed
        (e.g. a holiday falling on a Sunday being observed the following Monday).
        False may not work for all countries.

    Args:
        country: ISO 3166-1 alpha-2 code, e.g. AU
        states: states in the country e.g. ACT (default), NSW, NT, QLD, SA, TAS, VIC, WA in AU
        years: list of years to get holidays from. If None, empty dict is returned
    """
    assert years is not None and len(years) > 0, "years is required"

    result: Dict[datetime.datetime, str] = dict()
    if states is not None and len(states) > 0:
        for state in states:
            result |= holidays.utils.country_holidays(
                country=country,
                subdiv=state,
                years=years
            )
    else:   # country common holidays
        result = holidays.utils.country_holidays(
            country=country,
            years=years
        )

    return result


def is_holiday(
        target: Union[datetime.datetime, datetime.date, str],
        country: str = None,
        state: str = None
) -> bool:
    """Check if the target is holiday in the country or (country, state)
    Args:
        target: datetime, date, or date string to check
        country: country to check the holidays
        state: optional state of the country
    Returns: bool
    """
    assert \
        isinstance(target, str) or \
        isinstance(target, datetime.datetime) or \
        isinstance(target, datetime.date)

    year: int
    if isinstance(target, str):
        dates: List[datetime.datetime] = get_dates_from_string(target)
        assert len(dates) == 1, f"invalid date expression [{target}]"
        year = dates[0].year
    else:
        year = target.year

    return target in get_holidays(country=country, states=state, years=[year])


def is_weekend(
        target: Union[datetime.datetime, datetime.date, str],
) -> bool:
    """Check if the target is weekend
    Args:
        target: datetime, date, or date string to check
    Returns: bool
    """
    result: bool = False
    assert \
        isinstance(target, str) or \
        isinstance(target, datetime.datetime) or \
        isinstance(target, datetime.date)

    if isinstance(target, str):
        dates: List[datetime.datetime] = get_dates_from_string(target)
        assert len(dates) == 1, f"invalid date expression [{target}]"
        result = dates[0].weekday() in [WEEKDAY_SAT, WEEKDAY_SUN]
    else:
        result = target.weekday() in [WEEKDAY_SAT, WEEKDAY_SUN]

    return result


def get_cyclic_time_of_day(hours: int, minutes: int, seconds: int) -> Tuple[TYPE_FLOAT, TYPE_FLOAT]:
    """Encode time in day as cyclic
    https://datascience.stackexchange.com/a/6335/68313
    http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

    Args:
        hours: hours in day
        minutes: minutes in the hour
        seconds: seconds in the minute
    Returns: (time in sin, time in cos)
    """
    seconds_in_day = TYPE_FLOAT(24 * 60 * 60)
    assert 0 <= hours < 24, f"invalid hours [{hours}]"
    assert 0 <= hours < 60, f"invalid minutes [{minutes}]"
    assert 0 <= hours < 60, f"invalid seconds [{seconds}]"

    seconds: TYPE_FLOAT = \
        TYPE_FLOAT(60) * TYPE_FLOAT(60) * TYPE_FLOAT(hours) + \
        TYPE_FLOAT(60) * TYPE_FLOAT(minutes) + \
        TYPE_FLOAT(seconds)

    time_in_sin: TYPE_FLOAT
    time_in_cos: TYPE_FLOAT
    time_in_sin = x = np.sin(TYPE_FLOAT(2) * TYPE_FLOAT(np.pi) * seconds / seconds_in_day)
    time_in_cos = y = np.cos(TYPE_FLOAT(2) * TYPE_FLOAT(np.pi) * seconds / seconds_in_day)

    logger.debug("get_cyclic_time_of_day(): x/sin=[%s] y/cos=[%s]", time_in_sin, time_in_cos)
    return x, y


def get_cyclic_day_of_week(day_of_week: int) -> Tuple[TYPE_FLOAT, TYPE_FLOAT]:
    """Encode day of week as cyclic.
    day_of_week is an integer, where Monday is 0 and Sunday is 6.
    See https://docs.python.org/3/library/datetime.html#datetime.datetime.weekday

    https://datascience.stackexchange.com/a/6335/68313
    http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

    Args:
        day_of_week: day in week (1-7) where 1 = Sunday
    Returns: (day_of_week in sin, day_of_week in cos)
    """
    days_in_week = TYPE_FLOAT(7)
    assert 0 <= day_of_week < 7, f"invalid day_of_week [{day_of_week}]. Monday is 0 and Sunday is 6"

    day_of_week_in_sin: TYPE_FLOAT
    day_of_week_in_cos: TYPE_FLOAT

    day_of_week_in_sin = x = np.sin(TYPE_FLOAT(2) * TYPE_FLOAT(np.pi) * day_of_week / days_in_week)
    day_of_week_in_cos = y = np.cos(TYPE_FLOAT(2) * TYPE_FLOAT(np.pi) * day_of_week / days_in_week)

    logger.debug("get_cyclic_day_of_week(): x/sin=[%s] y/cos=[%s]", day_of_week_in_sin, day_of_week_in_cos)
    return x, y


def get_cyclic_month_of_year(months: int) -> Tuple[TYPE_FLOAT, TYPE_FLOAT]:
    """Encode month in year as cyclic and shift the values down by one such that
    it extends from 0 to 11, for convenience

    https://datascience.stackexchange.com/a/6335/68313
    http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

    Args:
        months: month in year (1-12)
    Returns: (month in sin, month in cos)
    """
    months_in_year = TYPE_FLOAT(12)
    assert 1 <= months <= 12, f"invalid months [{months}]"

    month_in_sin: TYPE_FLOAT
    month_in_cos: TYPE_FLOAT

    month_in_sin = x = np.sin(TYPE_FLOAT(2) * TYPE_FLOAT(np.pi) * (months - 1) / months_in_year)
    month_in_cos = y = np.cos(TYPE_FLOAT(2) * TYPE_FLOAT(np.pi) * (months - 1) / months_in_year)

    logger.debug("get_cyclic_month_of_year(): x/sin=[%s] y/cos=[%s]", month_in_sin, month_in_cos)
    return x, y
