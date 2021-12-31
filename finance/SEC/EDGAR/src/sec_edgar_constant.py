import os

NUM_CPUS = 8
FS_TYPE_10K = "10-K"
FS_TYPE_10Q = "10-Q"
EDGAR_BASE_URL = "https://sec.gov/Archives"
EDGAR_HTTP_HEADERS = {"User-Agent": "Company Name myname@company.com"}
DEFAULT_LOG_LEVEL = 20  # INFO


DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_INDEX = os.path.realpath(f"{DIR}/../data/index")
DATA_DIR_LISTING = os.path.realpath(f"{DIR}/../data/listing")
DATA_DIR_XBRL = os.path.realpath(f"{DIR}/../data/XBRL")

