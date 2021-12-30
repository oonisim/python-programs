#!/usr/bin/env python
"""
## Objective
Navigate through the EDGAR XBRL listings to generate the URLs to XBRL XML files.

## XBRL master index file
https://www.sec.gov/Archives/edgar/full-index stores the master index files.
[xbrl.gz] master index file for  TXT file in XBRL version

[Format]
|CIK    |Company Name|Form Type|Filing Date|TXT Path                                   |
|-------|------------|---------|-----------|-------------------------------------------|
|1002047|NetApp, Inc.|10-Q     |2020-02-18 |edgar/data/1002047/0001564590-20-005025.txt|

[Background]
Previously, XBRL was not introduced or not mandatory, hence the documents are
in HTML format and the TXT is the all-in-one including all the filing data
sectioned by <DOCUMENT>.

After the introduction of XBRL and recent mandates, the XBRL master index points
to the TXT file which is al-in-one including all the filing data in XML format. 
However, it is not straight-forward to parse the TXT for XBRL and there is
.xml file which has all the XML data only.

Hence, need to identify the URL to XBRL XML. However, the format fo the filename
is not consistent, e.g. <name>_htm.xml, <name>.xml. Needs to identify the XML
filename, then create the UR.

## EDGAR Directory Listing
Each filing directory provides the listing which lists all the files in the
directory either as index.html, index.json, or index.xml. Parse the index.xml
to identify the XBRL XML file.

https://sec.gov/Archives/edgar/full-index/${YEAR}/${QTR}/${ACCESSION}/index.xml

# EDGAR
* Investopedia -[Where Can I Find a Company's Annual Report and Its SEC Filings?]
(https://www.investopedia.com/ask/answers/119.asp)

> If you want to dig deeper and go beyond the slick marketing version of the annual
report found on corporate websites, you'll have to search through required filings
made to the Securities and Exchange Commission. All publicly-traded companies in
the U.S. must file regular financial reports with the SEC. These filings include
the annual report (known as the 10-K), quarterly report (10-Q), and a myriad of
other forms containing all types of financial data.45

# Quarterly filing indices
* [Accessing EDGAR Data](https://www.sec.gov/os/accessing-edgar-data)

> Using the EDGAR index files  
Indexes to all public filings are available from 1994Q3 through the present and
located in the following browsable directories:
> * https://www.sec.gov/Archives/edgar/daily-index/ — daily index files through the current year; (**DO NOT forget the trailing slash '/'**)
> * https://www.sec.gov/Archives/edgar/full-index/ — full indexes offer a "bridge" between quarterly and daily indexes, compiling filings from the beginning of the current quarter through the previous business day. At the end of the quarter, the full index is rolled into a static quarterly index.
> 
> Each directory and all child sub directories contain three files to assist in automated crawling of these directories. Note that these are not visible through directory browsing.
> * index.html (the web browser would normally receive these)
> * index.xml (an XML structured version of the same content)
> * index.json (a JSON structured vision of the same content)
> 
> Four types of indexes are available:
> * company — sorted by company name
> * form — sorted by form type
> * master — sorted by CIK number 
> * **XBRL** — list of submissions containing XBRL financial files, sorted by CIK number; these include Voluntary Filer Program submissions
> 
> The EDGAR indexes list the following information for each filing:
> * company name
> * form type
> * central index key (CIK)
> * date filed
> * file name (including folder path)

## Example

Full index files for 2006 QTR 3.
<img src="../image/edgar_full_index_quarter_2006QTR3.png" align="left" width="800"/>
"""
# ================================================================================
# Setup
# ================================================================================
from typing import (
    List,
    Dict,
    Iterable
)
import argparse
import sys
import os
import pathlib
import logging
import glob
import re
import json
import time
import requests
import ray
import bs4
import pandas as pd
from bs4 import (
    BeautifulSoup
)
import Levenshtein as levenshtein
from sec_edgar_constant import (
    NUM_CPUS,
    FS_TYPE_10K,
    FS_TYPE_10Q,
    EDGAR_BASE_URL,
    EDGAR_HTTP_HEADERS,
    DEFAULT_LOG_LEVEL,
    DATA_DIR_LISTING,
    DATA_DIR_XBRL,
)
from sec_edgar_utility import(
    split,
    http_get_content,
)


pd.set_option('display.max_colwidth', None)

# --------------------------------------------------------------------------------
# TODO:
#  Logging setup for Ray as in https://docs.ray.io/en/master/ray-logging.html.
#  In Ray, all of the tasks and actors are executed remotely in the worker processes.
#  Since Python logger module creates a singleton logger per process, loggers should
#  be configured on per task/actor basis.
# --------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger(__name__)
# Logger.addHandler(logging.StreamHandler())


# ================================================================================
# Utilities
# ================================================================================
def get_command_line_arguments():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description='argparse test program')
    parser.add_argument(
        '-i', '--input-directory', type=str, required=False, default=DATA_DIR_LISTING,
        help='specify the input data directory'
    )
    parser.add_argument(
        '-o', '--output-directory', type=str, required=False, default=DATA_DIR_XBRL,
        help='specify the output data directory'
    )
    parser.add_argument(
        '-y', '--year', type=int, required=False, help='specify the target year'
    )
    parser.add_argument(
        '-q', '--qtr', type=int, choices=[1, 2, 3, 4], required=False,
        help='specify the target quarter'
    )
    parser.add_argument(
        '-n', '--num-workers', type=int, required=False,
        help='specify the number of workers to use'
    )
    parser.add_argument(
        '-l', '--log-level', type=int, choices=[10, 20, 30, 40], required=False,
        help='specify the logging level (10 for DEBUG, 20 for INFO)',
    )
    args = vars(parser.parse_args())

    # Mandatory
    input_directory = args['input_directory']
    output_directory = args['output_directory']

    # Optional
    year = str(args['year']) if args['year'] else None
    qtr = str(args['qtr']) if args['qtr'] else None
    num_workers = args['num_workers'] if args['num_workers'] else NUM_CPUS
    log_level = args['log_level'] if args['log_level'] else DEFAULT_LOG_LEVEL

    return input_directory, output_directory, year, qtr, num_workers, log_level


def xbrl_xml_file_path_to_save(directory, filename):
    """Generate the file path to save the XBRL XML. Create directory if required.
    Args:
        directory: location to save the file
        flilename: name of the file to create
    """
    pathlib.Path(directory).mkdir(mode=0o775, parents=True, exist_ok=True)
    setattr(xbrl_xml_file_path_to_save, "mkdired", True)

    destination = f"{directory}/{filename}"
    return destination


def list_files(input_directory, output_directory, year=None, qtr=None):
    """List files in the directory
    When year is specified, only matching listing files for the year will be selected.
    When qtr is specified, only match8ng listing files for the quarter will be selected.

    Args:
        input_directory: path to the directory from where to get the file
        output_directory: path to the directory where XML is saved
        year: Year to select
        qtr: Quarter to select
    Returns: List of flies
    """
    assert os.path.isdir(input_directory), f"Not a directory or does not exist: {input_directory}"
    assert (re.match(r"[1-2][0-9][0-9][0-9]", year) if year else True), f"Invalid year {year}"
    assert (re.match(r"[1-4]", qtr) if qtr else True), f"Invalid quarter {qtr}"

    def is_file_to_process(filepath):
        """Verify if the filepath points to a file that has not been processed yet.
        If XBRL file has been already created and exists, then skip the filepath.
        """
        if os.path.isfile(filepath) and os.access(filepath, os.R_OK):
            Logger.info("list_files(): adding [%s] to handle" % filepath)
            return True
        else:
            Logger.info("[%s] does not exist, cannot read, or not a file. skipping." % filepath)
            return False

    pattern = ""
    pattern += f"{year}" if year else "*"
    pattern += "QTR"
    pattern += f"{qtr}" if qtr else "?"
    pattern += "_XBRL.gz"

    Logger.info("Listing the files to process in the directory %s ..." % input_directory)
    files = sorted(filter(is_file_to_process, glob.glob(input_directory + os.sep + pattern)))
    Logger.info("No files to process in the directory %s" % input_directory)
    return files


def save_to_xml(content, directory, filename):
    """Save the XBRL XML"""
    destination = xbrl_xml_file_path_to_save(directory=directory, filename=filename)
    Logger.debug("save_to_xml(): saving XML to [%s]..." % destination)
    with open(destination, "w") as f:
        f.write(content)


# ================================================================================
# Logic
# ================================================================================
def get_xml(url, logger):
    """GET XBRL XML from the URL
    Args:
        url: URL to download the XBRL XML
        logger: Logging instance to use
    Returns: XBRL XML content or None
    """
    max_retries_allowed = 3
    while True:
        try:
            # --------------------------------------------------------------------------------
            # https://www.sec.gov/oit/announcement/new-rate-control-limits
            # If a user or application submits more than 10 requests per second to EDGAR websites,
            # SEC may limit further requests from the relevant IP address(es) for a brief period.
            #
            # TODO:
            #  Synchronization among workers to limit the rate 10/sec from the same IP.
            #  For now, just wait 1 sec at each invocation from the worker.
            # --------------------------------------------------------------------------------
            # time.sleep(1)

            content = http_get_content(url, EDGAR_HTTP_HEADERS)
            logger.debug("get_xml(): got content from [%s]" % url)
            return content
        except RuntimeError as e:
            max_retries_allowed -= 1
            if max_retries_allowed > 0:
                logger.error("get_xml(): failed to get [%s]. retrying..." % url)
                time.sleep(30)
            else:
                logger.error("get_xml(): failed to get [%s]. skipping..." % url)
                break

    return None


@ray.remote(num_returns=1)
def worker(df, log_level=logging.INFO):
    """GET XBRL XML and save to a file.
    Args:
        df: Pandas dataframe |CIK|Company Name|Form Type|Date Filed|Filename|
            where 'Filename' is the URL to index.xml in the filing directory.
        log_level: Logging level to use in the worker.
    Returns: List of indices of the dataframe that have failed to get XBRL XML.
    """
    assert df is not None and len(df) > 0, "worker(): invalid dataframe"

    # --------------------------------------------------------------------------------
    #  Logging setup for Ray as in https://docs.ray.io/en/master/ray-logging.html.
    #  In Ray, all of the tasks and actors are executed remotely in the worker processes.
    #  Since Python logger module creates a singleton logger per process, loggers should
    #  be configured on per task/actor basis.
    # --------------------------------------------------------------------------------
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)
    logger.info("worker(): task size is %s" % len(df))

    failed_indices = []
    for index, row in df.iterrows():
        # --------------------------------------------------------------------------------
        # Download XBRL XML
        # --------------------------------------------------------------------------------
        url = row['Filename']
        content = get_xml(url, logger)

        if content:
            # --------------------------------------------------------------------------------
            # Save XML
            # URL format: https://sec.gov/Archives/edgar/data/{cik}}/{accession}}/{filename}
            # https://sec.gov/Archives/edgar/data/1000697/000095012310017583/wat-20091231.xml
            # --------------------------------------------------------------------------------
            elements = url.split('/')
            filename = elements[-1]
            accession = elements[-2]
            cik = elements[-3]
            assert str(row['CIK']) == cik, \
                f"CIK [{row['CIK']})] must match CIK part [{cik}] in url {url}"

            directory = f"{DATA_DIR_XBRL}{os.sep}{cik}{os.sep}{accession}"
            logger.debug(f"worker(): saving XML to [{directory}:{filename}]...")

            save_to_xml(content, directory, filename)
        else:
            failed_indices.append(index)

    return failed_indices


def director(df, num_workers: int):
    """Director to download XBRL XML files from the SEC filing directories.
    Args:
        df: pandas datafrome of XBRL indices
        num_workers: Number of workers to use
    Returns: Pandas dataframe of failed records
    """
    # --------------------------------------------------------------------------------
    # Split dataframe to handle in parallel
    # --------------------------------------------------------------------------------
    assignment = split(tasks=df, num=num_workers)

    # --------------------------------------------------------------------------------
    # Asynchronously invoke tasks
    # --------------------------------------------------------------------------------
    futures = [worker.remote(task) for task in assignment]
    assert len(futures) == num_workers, f"Expected {num_workers} tasks but got {len(futures)}."

    # --------------------------------------------------------------------------------
    # Wait for the job completion
    # --------------------------------------------------------------------------------
    waits = []
    while futures:
        completed, futures = ray.wait(futures)
        waits.extend(completed)

    # --------------------------------------------------------------------------------
    # Collect the results
    # --------------------------------------------------------------------------------
    assert len(waits) == num_workers, f"Expected {num_workers} tasks but got {len(waits)}."
    failed_indices = sum(ray.get(waits), [])

    return df.loc[failed_indices, :]


def load_edgar_xbrl_listing_file(filepath, types=[FS_TYPE_10K, FS_TYPE_10K]):
    """Generate a pandas dataframe for each XBRL listing file.
    Args:
        filepath: path to a XBRL index file
        types: Filing types e.g. 10-K
    Returns:
        pandas dataframe
    """
    Logger.info("load_edgar_xbrl_listing_file(): filepath [%s]" % filepath)
    assert os.path.isfile(filepath) and os.access(filepath, os.R_OK), \
        f"{filepath} is not a file, cannot read, or does not exist."

    # --------------------------------------------------------------------------------
    # Load XBRL index CSV file
    # --------------------------------------------------------------------------------
    listings = pd.read_csv(
        filepath,
        skip_blank_lines=True,
        header=0,         # The 1st data line after omitting skiprows and blank lines.
        sep='|',
        usecols=['CIK', 'Form Type', 'Filename'],
        # parse_dates=['Date Filed'],
    )

    # --------------------------------------------------------------------------------
    # Select filing for the target Form Types e.g. 10-K
    # --------------------------------------------------------------------------------
    listings = listings.loc[listings['Form Type'].isin(types)] if types else listings
    listings.loc[:, 'Form Type'] = listings['Form Type'].astype('category')
    Logger.info("load_edgar_xbrl_listing_file(): size of listings [%s]" % len(listings))

    return listings


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------------------------
    # Command line arguments
    # --------------------------------------------------------------------------------
    args = get_command_line_arguments()
    input_directory, output_directory, year, qtr, num_workers, log_level = args

    # --------------------------------------------------------------------------------
    # XBRL XML logic
    # --------------------------------------------------------------------------------
    try:
        # --------------------------------------------------------------------------------
        # Setup Ray
        # --------------------------------------------------------------------------------
        Logger.info("main(): initializing Ray using %s workers..." % num_workers)
        ray.init(num_cpus=num_workers, num_gpus=0, logging_level=log_level)

        for filepath in list_files(
            input_directory=input_directory, output_directory=output_directory, year=year, qtr=qtr
        ):
            filename = os.path.basename(filepath)
            Logger.info("main(): processing the listing [%s]..." % filename)

            failed_df = director(
                df=load_edgar_xbrl_listing_file(filepath=filepath, types=[FS_TYPE_10Q, FS_TYPE_10K]),
                num_workers=num_workers
            )
            if len(failed_df):
                Logger.info("main(): failed records for %s\n%s" % (filename, failed_df))
            else:
                Logger.info("main(): all records processed in %s" % filename)
    finally:
        # --------------------------------------------------------------------------------
        # Clean up resource
        # --------------------------------------------------------------------------------
        Logger.info("main(): shutting down Ray...")
        ray.shutdown()

