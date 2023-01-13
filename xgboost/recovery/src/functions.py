"""
Main module
"""
import math
import sys
import logging
import datetime
import pandas as pd
import numpy as np

from util_constant import (
    TYPE_FLOAT,
)
from util_datetime import (
    convert_date_into_datetime,
    convert_time_into_timedelta,
    convert_date_time_into_datetime,
    parse_time_string,
    has_date_in_string,
    parse_datetime_string,
    is_holiday,
    is_weekend,
    get_cyclic_time_of_day,
    get_cyclic_day_of_week,
)

from constants import (
    COUNTRY_CODE,
    COLUMN_START_TIME,
    COLUMN_END_TIME,
    COLUMN_WEEKDAY_SINX,
    COLUMN_WEEKDAY_COSY,
)

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logger = logging.getLogger("analysis")


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
def get_start_date_time(row: pd.Series) -> datetime.datetime:
    name: str = "get_start_date_time()"
    logger.debug("%s: row type[%s] row[%s]", name, type(row), row)

    # --------------------------------------------------------------------------------
    # if timeStart column is not valid, the row is invalid as there should be no way
    # to recover start time.
    # --------------------------------------------------------------------------------
    if pd.isnull(row['timeStart']):
        return np.nan

    start_date_time: datetime.datetime = np.nan

    # Begin start_date_time extraction
    if has_date_in_string(row['timeStart']):
        # --------------------------------------------------------------------------------
        # timeStart already includes date.
        # --------------------------------------------------------------------------------
        start_date_time = parse_datetime_string(row['timeStart'])
        logger.debug("%s: start_date_time is [%s]", name, start_date_time)

    else:
        # --------------------------------------------------------------------------------
        # timeStart has no date, then (date, startTime) columns must be valid.
        # Otherwise the row is invalid.
        # --------------------------------------------------------------------------------
        if pd.isnull(row['date']):
            # --------------------------------------------------------------------------------
            # Invalid row. Return NaN as the start_date_time to mark the row as invalid.
            # --------------------------------------------------------------------------------
            start_date_time = np.nan

        else:
            # --------------------------------------------------------------------------------
            # Date from 'date' column, and omit the time part.
            # --------------------------------------------------------------------------------
            _date_time: datetime.datetime = parse_datetime_string(row['date'])
            assert isinstance(_date_time, datetime.datetime)

            _date: datetime.date = _date_time.date()
            logger.debug("%s: date is [%s]", name, _date)

            # --------------------------------------------------------------------------------
            # Time from 'timeStart' column
            # --------------------------------------------------------------------------------
            _temp_date_time: datetime.datetime = parse_datetime_string(row['timeStart'])
            assert isinstance(_temp_date_time, datetime.datetime)

            _time: datetime.time = _temp_date_time.time()
            logger.debug("%s: start_time is [%s]", name, _time)

            start_date_time = convert_date_time_into_datetime(_date, _time)
            logger.debug("%s: start_date_time is [%s]", name, start_date_time)

    # End start_date_time extraction

    return start_date_time


def get_end_date_time(row: pd.Series) -> datetime.datetime:
    name: str = "get_end_date_time()"
    logger.debug("%s: row type[%s] row[%s]", name, type(row), row)

    assert isinstance(row[COLUMN_START_TIME], datetime.datetime), \
        f"expected start_date_time as datetime.datetime, got [{type(row[COLUMN_START_TIME])}]."
    start_date_time: datetime.datetime = row[COLUMN_START_TIME]

    # --------------------------------------------------------------------------------
    # if timeEnd and processTime columns are not valid, the row is invalid
    # --------------------------------------------------------------------------------
    if pd.isnull(row['timeEnd']) and pd.isnull(row['processTime']):
        return np.nan

    # Begin end_date_time extraction
    end_date_time: datetime.datetime = np.nan

    if not pd.isnull(row['timeEnd']):
        # --------------------------------------------------------------------------------
        # Acquire end_date_time from (start_date_time, timeEnd)
        # --------------------------------------------------------------------------------
        if has_date_in_string(row['timeEnd']):
            # --------------------------------------------------------------------------------
            # timeEnd already includes date.
            # --------------------------------------------------------------------------------
            end_date_time = parse_datetime_string(row['timeEnd'])
            logger.debug("%s: end_date_time is [%s]", name, end_date_time)

        else:
            # --------------------------------------------------------------------------------
            # timeEnd has no date, hence it is only time expression, and date is from start_time.
            # --------------------------------------------------------------------------------
            _dummy_end_date_time: datetime.datetime = parse_datetime_string(row['timeEnd'])
            assert isinstance(_dummy_end_date_time, datetime.datetime)

            # take only time part
            _time: datetime.time = _dummy_end_date_time.time()
            logger.debug("%s: timeEnd is [%s]", name, _time)

            assert isinstance(row[COLUMN_START_TIME], datetime.datetime)
            _date = row[COLUMN_START_TIME].date()

            end_date_time = convert_date_time_into_datetime(
                date_in_year=_date,
                time_in_day=_time
            )

            # Make sure timeEnd is after start_time in case the processing is crossing midnight.
            if end_date_time <= start_date_time:
                logger.warning(
                    "%S: end_date_time [%s] is before start_date_time [%s]",
                    end_date_time, start_date_time
                )
                # Advance the end_date_time with 24h.
                end_date_time = end_date_time + datetime.timedelta(days=1)
                logger.debug("%s: end_date_time is [%s]", name, end_date_time)

    else:
        # --------------------------------------------------------------------------------
        # Acquire end_date_time from (start_date_time, processTime)
        # [Assumption] processTime is hh:mm:yy and mm is less than 60.
        # --------------------------------------------------------------------------------
        try:
            assert isinstance(row['processTime'], str), \
                f"expected row['processTime'] of type str, got [{type(row['processTime'])}]"

            _time: datetime.time = parse_time_string(row['processTime'])
            delta: datetime.timedelta = convert_time_into_timedelta(time_in_day=_time)
            end_date_time = start_date_time + delta

        except ValueError as e:
            logging.error("%s: invalid time expression [%s]", name, row['processTime'])
            end_date_time = np.nan

    # End start_date_time extraction

    return end_date_time


def get_process_time(row: pd.Series) -> pd.Series:
    """Get proces time in seconds as TYPE_FLOAT
    Process time as end time - start time.

    Return: process time taken in second (as TYPE_FLOAT)
    """
    assert isinstance(row[COLUMN_START_TIME], datetime.datetime)
    assert isinstance(row[COLUMN_END_TIME], datetime.datetime)

    delta: datetime.timedelta = row[COLUMN_END_TIME] - row[COLUMN_START_TIME]
    assert isinstance(delta, datetime.timedelta), \
        f"expected delta as type datetime.timedelta, got {delta} of type [type(delta)]"

    return TYPE_FLOAT(delta.total_seconds())


def get_process_date(row: pd.Series) -> int:
    _date_time: datetime.datetime = np.nan

    if not pd.isnull(row[COLUMN_START_TIME]):
        _date_time = convert_date_into_datetime(row[COLUMN_START_TIME].date())

    return _date_time.day


def get_code_from_supplier(supplier: str):
    supplier_to_code = {
        "mary therese": "mat",
        "mary": "mar",
        'mary anne': "maa",
        'mary jane': 'maj',
        'dick tracey': 'dic',
        'tom hanks': 'tom',
        'harry houdini': 'har',
    }
    return supplier_to_code.get(supplier.lower(), np.nan)


def get_numeric_supplier_code(code: str):
    str_to_num = {
        "mat": 0,
        "mar": 1,
        "maa": 2,
        "maj": 3,
        "har": 4,
        "dic": 5,
        "tom": 6,
    }
    return str_to_num.get(code.lower(), np.nan)


def get_supplier_code(row: pd.Series):
    code: str
    if isinstance(row['supplierCode'], str):
        code = row['supplierCode']
    else:
        code = get_code_from_supplier(row['supplier'])

    return get_numeric_supplier_code(code)


def get_numeric_facility_code(code: str):
    str_to_num = {
        "newcastle": 0,
        "bundaberg": 1,
    }
    return str_to_num.get(code.lower(), np.nan)


def get_facility_code(row: pd.Series):
    code: str
    # if row['facility'] not in (np.nan, math.nan, None):
    if isinstance(row['facility'], str):
        code = row['facility']
    else:
        code = "n/a"

    return get_numeric_facility_code(code)


def get_weekday(row: pd.Series):
    assert isinstance(row[COLUMN_START_TIME], datetime.datetime), \
        f"expected [{COLUMN_START_TIME}] as timedate.timedate, got [{type(datetime.datetime)}]"

    date_time: datetime.datetime = row[COLUMN_START_TIME]
    return date_time.weekday()  # 0: Mon, 6: Sun


def get_weekday_cyclic(row: pd.Series) -> pd.Series:
    """Hour of the start time in cyclic (sin_as_x, cos_as_y)"""
    assert isinstance(row[COLUMN_START_TIME], datetime.datetime), \
        f"expected [{COLUMN_START_TIME}] as timedate.timedate, got [{type(datetime.datetime)}]"

    x: TYPE_FLOAT
    y: TYPE_FLOAT

    date_time: datetime.datetime = row[COLUMN_START_TIME]
    x, y = get_cyclic_day_of_week(date_time.weekday())
    return pd.Series([x, y],index=[COLUMN_WEEKDAY_SINX, COLUMN_WEEKDAY_COSY])


def get_holiday(row: pd.Series):
    """Flag if the start date/time is on holiday (1: holiday, 0: not holiday)"""
    assert isinstance(row[COLUMN_START_TIME], datetime.datetime), \
        f"expected [{COLUMN_START_TIME}] as timedate.timedate, got [{type(datetime.datetime)}]"

    date_time: datetime.datetime = row[COLUMN_START_TIME]
    return int(is_holiday(target=date_time, country=COUNTRY_CODE) or is_weekend(target=date_time))


def get_start_hour(row: pd.Series):
    """Hour of the start time"""
    assert isinstance(row[COLUMN_START_TIME], datetime.datetime), \
        f"expected [{COLUMN_START_TIME}] as timedate.timedate, got [{type(datetime.datetime)}]"

    date_time: datetime.datetime = row[COLUMN_START_TIME]
    return int(date_time.hour)


def get_start_time_cyclic(row: pd.Series) -> pd.Series:
    """Hour of the start time in cyclic (sin_as_x, cos_as_y)"""
    assert isinstance(row[COLUMN_START_TIME], datetime.datetime), \
        f"expected [{COLUMN_START_TIME}] as timedate.timedate, got [{type(datetime.datetime)}]"

    x: TYPE_FLOAT
    y: TYPE_FLOAT

    date_time: datetime.datetime = row[COLUMN_START_TIME]
    x, y = get_cyclic_time_of_day(
        hours=date_time.hour,
        minutes=date_time.minute,
        seconds=date_time.second
    )
    return pd.Series([x, y], index=[COLUMN_WEEKDAY_SINX, COLUMN_WEEKDAY_COSY])


