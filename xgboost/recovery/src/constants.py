# --------------------------------------------------------------------------------
# Global constant
# --------------------------------------------------------------------------------
# Location
COUNTRY_CODE: str = "AU"

# Actors
COLUMN_FACILITY_CODE: str = "facility_code"
COLUMN_FACILITY: str = 'facility'
COLUMN_SUPPLIER_CODE: str = 'supplier_code'
# Timing
COLUMN_PROCESS_DATE: str = 'process_date'
COLUMN_START_TIME: str = 'start_date_time'
COLUMN_START_HOUR: str = "start_hour"
COLUMN_START_TIME_SINX: str = "start_time_sin_x"
COLUMN_START_TIME_COSY: str = "start_time_cos_y"
COLUMN_END_TIME: str = "end_date_time"
COLUMN_WEEKDAY: str = "dayofweek"
COLUMN_WEEKDAY_SINX: str = "dayofweek_sin_x"
COLUMN_WEEKDAY_COSY: str = "dayofweek_cos_y"
COLUMN_IS_HOLIDAY: str = 'is_holiday'  # holiday or weenends
# Performance
COLUMN_PROCESS_TIME: str = "process_time"
COLUMN_INPUT: str = "input"
COLUMN_OUTPUT: str = "output"
COLUMN_THROUGHPUT: str = 'throughput'
COLUMN_RECOVERY_RATE: str = 'recovery_rate'

# Utility
DEBUG: bool = False
SECONDS_IN_MIN = 60
