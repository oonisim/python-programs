import os

NUM_CPUS = 8
FS_TYPE_10K = "10-K"
FS_TYPE_10Q = "10-Q"
EDGAR_BASE_URL = "https://sec.gov/Archives"
EDGAR_HTTP_HEADERS = {"User-Agent": "Company Name myname@company.com"}
DEFAULT_LOG_LEVEL = 20  # INFO


DIR = os.path.dirname(os.path.realpath(__file__))
DIR_CSV_INDEX = os.path.realpath(f"{DIR}/../data/csv/index")
DIR_CSV_LIST = os.path.realpath(f"{DIR}/../data/csv/listing")
DIR_CSV_XBRL = os.path.realpath(f"{DIR}/../data/csv/xbrl")
DIR_XML_XBRL = os.path.realpath(f"{DIR}/../data/xml/xbrl")

