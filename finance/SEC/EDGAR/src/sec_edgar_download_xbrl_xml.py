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
import os
import pathlib
import logging
import glob
import re
import time
import random
import gzip

import ray
import pandas as pd
from sec_edgar_constant import (
    NUM_CPUS,
    FS_TYPE_10K,
    FS_TYPE_10Q,
    EDGAR_HTTP_HEADERS,
    DEFAULT_LOG_LEVEL,
    DIR_CSV_LIST,
    DIR_CSV_XBRL,
    DIR_XML_XBRL,
)
from sec_edgar_utility import(
    split,
    http_get_content,
)


pd.set_option('display.max_colwidth', None)


# ================================================================================
# Utilities
# ================================================================================
def get_command_line_arguments():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description='argparse test program')
    parser.add_argument(
        '-ic', '--input-csv-directory', type=str, required=False, default=DIR_CSV_LIST,
        help='specify the input data directory'
    )
    parser.add_argument(
        '-oc', '--output-csv-directory', type=str, required=False, default=DIR_CSV_XBRL,
        help='specify the output data directory to save the csv file (not xml)'
    )
    parser.add_argument(
        '-ox', '--output-xml-directory', type=str, required=False, default=DIR_XML_XBRL,
        help='specify the output data directory to save the xml file (not csv)'
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
        default=DEFAULT_LOG_LEVEL,
        help='specify the logging level (10 for INFO)',
    )
    args = vars(parser.parse_args())

    # Mandatory
    input_csv_directory = args['input_csv_directory']
    output_csv_directory = args['output_csv_directory']
    output_xml_directory = args['output_xml_directory']

    # Optional
    year = str(args['year']) if args['year'] else None
    qtr = str(args['qtr']) if args['qtr'] else None
    num_workers = args['num_workers'] if args['num_workers'] else NUM_CPUS
    log_level = args['log_level'] if args['log_level'] else DEFAULT_LOG_LEVEL

    return input_csv_directory, output_csv_directory, output_xml_directory, year, qtr, num_workers, log_level


def xml_relative_path_to_save(directory, basename):
    """
    Generate the relative file path from the output_xml_directory to save the XBRL XML.
    XML is saved to {output_xml_directory}/{directory}/{basename}.gz.
    The function returns {directory}/{basename}.gz part.

    Args:
        directory: location to save the file
        basename: basename of the file to save
    Returns: Absolute file path
    """
    assert not directory.startswith(os.sep), "Must be relative directory path"
    relative = f"{directory}{os.sep}{basename}.gz"
    return relative


def xml_absolute_path_to_save(output_xml_directory, directory, basename):
    f"""
    Generate the absolute file path to save the XBRL XML. Create directory if required.
    Each XML is saved to {output_xml_directory}/{directory}/{basename}.gz.

    Args:
        output_xml_directory: Base directory for XML
        directory: Relative path from the output_xml_directory
        basename: basename of the file to save
    Returns: Absolute file path
    """
    relative = xml_relative_path_to_save(directory, basename)
    absolute = os.path.realpath(f"{output_xml_directory}{os.sep}{relative}")
    pathlib.Path(os.path.dirname(absolute)).mkdir(mode=0o775, parents=True, exist_ok=True)
    return absolute


def csv_absolute_path_to_save(directory, basename):
    """
    Generate the absolute file path to save the dataframe as csv. Create directory if required.

    Args:
        directory: *Absolute* path to the directory to save the file
        basename: Name of the file to save

    Returns: Absolute file path
    """
    if not hasattr(csv_absolute_path_to_save, "mkdired"):
        pathlib.Path(directory).mkdir(mode=0o775, parents=True, exist_ok=True)
        setattr(csv_absolute_path_to_save, "mkdired", True)

    absolute = os.path.realpath(f"{directory}{os.sep}{basename}_XBRL.gz")
    return absolute


def list_csv_files(input_csv_directory, output_csv_directory, year=None, qtr=None):
    """List files in the directory
    When year is specified, only matching listing files for the year will be selected.
    When qtr is specified, only match8ng listing files for the quarter will be selected.

    Args:
        input_csv_directory: path to the directory from where to get the file
        output_csv_directory: path to the directory where XML is saved
        year: Year to select
        qtr: Quarter to select
    Returns: List of flies
    """
    assert os.path.isdir(input_csv_directory), f"Not a directory or does not exist: {input_csv_directory}"
    assert (isinstance(year, str) and re.match(r"^[1-2][0-9]{3}$", year) if year else True), \
        f"Invalid year {year} of type {type(year)}"
    assert (isinstance(qtr, str) and re.match(r"^[1-4]$", qtr) if qtr else True), \
        f"Invalid quarter {qtr} of type {type(qtr)}"

    def is_file_to_process(filepath):
        """Verify if the filepath points to a file that has not been processed yet.
        If XBRL file has been already created and exists, then skip the filepath.
        """
        if os.path.isfile(filepath) and os.access(filepath, os.R_OK):
            logging.info("list_csv_files(): adding [%s] to handle" % filepath)

            path_to_csv = csv_absolute_path_to_save(output_csv_directory, os.path.basename(filepath))
            if os.path.isfile(path_to_csv):
                logging.info(
                    "list_csv_files(): XBRL  management file [%s] already exits. skipping [%s]."
                    % (path_to_csv, filepath)
                )
                return False
            else:
                logging.info("list_csv_files(): adding [%s] to handle" % filepath)
                return True
        else:
            logging.info("[%s] does not exist, cannot read, or not a file. skipping." % filepath)
            return False

    pattern = ""
    pattern += f"{year}" if year else "*"
    pattern += "QTR"
    pattern += f"{qtr}" if qtr else "?"
    pattern += "_LIST.gz"

    logging.info("Listing the files to process in the directory %s ..." % input_csv_directory)
    files = sorted(
        filter(is_file_to_process, glob.glob(input_csv_directory + os.sep + pattern)),
        reverse=True
    )
    if files is None or len(files) == 0:
        logging.info("No files to process in the directory %s" % input_csv_directory)

    return files


def get_xml(url):
    """GET XBRL XML from the URL
    Args:
        url: URL to download the XBRL XML
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
            time.sleep(1)

            logging.info("get_xml(): getting XBRL XML [%s]..." % url)
            content = http_get_content(url, EDGAR_HTTP_HEADERS)
            return content
        except RuntimeError as e:
            max_retries_allowed -= 1
            if max_retries_allowed > 0:
                logging.error("get_xml(): failed to get [%s]. retrying..." % url)
                time.sleep(random.randint(30, 90))
            else:
                logging.error("get_xml(): failed to get [%s]. skipping..." % url)
                break

    return None


def save_to_xml(msg):
    """Save the XBRL XML to the path relative to DIR_XML_XBRL.

    Args:
        msg: message including the content data, directory to save it as filename
    Returns: relative path to the XML file if successfully saved, else None
    """
    content = msg["data"]
    output_xml_directory = msg['output_xml_directory']
    directory: str = msg['directory']
    basename: str = msg['basename']
    assert not directory.startswith("/"), f"Must not start with '/' but [{directory}]"

    destination = xml_absolute_path_to_save(
        output_xml_directory=output_xml_directory, directory=directory, basename=basename
    )
    extension = os.path.splitext(destination)[1]

    logging.debug("save_to_xml(): saving XBRL XML to [%s]..." % destination)
    try:
        if extension == ".gz":
            with gzip.open(f"{destination}", 'wb') as f:
                f.write(content.encode())
        elif extension == "":
            with open(destination, "w") as f:
                f.write(content.encode)
        else:
            assert False, f"Unknown file type [{extension}]"
    except IOError as e:
        logging.error("save_to_xml(): failed to save [%s] due to [%s]" % (destination, e))
        return None
    else:
        logging.debug("save_to_xml(): saved [%s]" % destination)
        return xml_relative_path_to_save(directory, basename)


def save_to_csv(msg):
    """Save the dataframe to CSV"""
    df = msg['data']
    directory: str = msg['directory']
    basename: str = msg['basename']
    compression: str = msg['compression']

    destination = csv_absolute_path_to_save(directory, basename)
    logging.info("save_to_csv(): saving dataframe to [%s]..." % destination)
    df.to_csv(
        destination,
        sep="|",
        compression=compression,
        header=True,
        index=False
    )


# ================================================================================
# Logic
# ================================================================================
@ray.remote(num_returns=1)
def worker(msg:dict):
    """
    1. GET XBRL XML from the filing directory and save to a file.
    2. Update the listing dataframe with the year, qtr, path to the saved XBRL XML.
       Set None to the path column when failed to get the XBRL XML.

    The incoming dataframe has the format where 'Filename' is the URL to XBRL XML
    in the filing directory.
    |CIK|Company Name|Form Type|Date Filed|Filename|

    Args:
        msg: Dictionary to data package of format {
                "data": <dataframe>,
                "year": <year of the filing>,
                "qtr": <quarter of the filing>,
                "log_level": <logging level>
        }

    Returns: Updated dataframe
    """
    df = msg["data"]
    year: str = msg['year']
    qtr: str = msg['qtr']
    output_xml_directory = msg["output_xml_directory"]
    log_level:int = msg['log_level']
    assert df is not None and len(df) > 0, f"Invalid dataframe \n{df}"

    # --------------------------------------------------------------------------------
    # Add year/qtr/filepath columns
    # --------------------------------------------------------------------------------
    df.insert(loc=3, column='Year', value=pd.Categorical([year]* len(df)))
    df.insert(loc=4, column='Quarter', value=pd.Categorical([qtr]* len(df)))
    df.insert(loc=len(df.columns), column="Filepath", value=[None]*len(df))

    # --------------------------------------------------------------------------------
    #  Logging setup for Ray as in https://docs.ray.io/en/master/ray-logging.html.
    #  In Ray, all of the tasks and actors are executed remotely in the worker processes.
    #  Since Python logger module creates a singleton logger per process, loggers should
    #  be configured on per task/actor basis.
    # --------------------------------------------------------------------------------
    logging.basicConfig(level=log_level)
    logging.info("worker(): task size is %s" % len(df))

    failed_indices = []
    for index, row in df.iterrows():
        # --------------------------------------------------------------------------------
        # Download XBRL XML
        # --------------------------------------------------------------------------------
        url = row['Filename']
        content = get_xml(url)

        if content:
            # --------------------------------------------------------------------------------
            # Save XBRL XML
            # URL format: https://sec.gov/Archives/edgar/data/{cik}}/{accession}}/{filename}
            # https://sec.gov/Archives/edgar/data/1000697/000095012310017583/wat-20091231.xml
            # --------------------------------------------------------------------------------
            elements = url.split('/')
            basename = elements[-1]
            accession = elements[-2]
            cik = elements[-3]
            assert str(row['CIK']) == cik, \
                f"CIK [{row['CIK']})] must match CIK part [{cik}] in url {url}"

            # Note: The directory is relative path, NOT absolute
            directory = f"{cik}{os.sep}{accession}"
            package = {
                "data": content,
                "output_xml_directory": output_xml_directory,
                "directory": directory,
                "basename": basename
            }
            path_to_saved_xml = save_to_xml(package)

            # --------------------------------------------------------------------------------
            # Update the dataframe with the filepath where the XBRL XML has been saved.
            # --------------------------------------------------------------------------------
            if path_to_saved_xml:
                df.at[index, 'Filepath'] = path_to_saved_xml
                logging.debug(
                    "worker(): updated the dataframe[%s, 'Filepath'] with [%s]."
                    % (index, path_to_saved_xml)
                )
            else:
                logging.debug(
                    "worker(): not updated the dataframe[%s, 'Filepath'] as saving has failed."
                    % index
                )
        else:
            logging.debug(
                "worker(): not updated the dataframe[%s, 'Filepath'] as failed to get XBRL XML"
                % index
            )

    return df


def director(msg):
    """Director to download XBRL XML files from the SEC filing directories.
    Args:
        df: pandas datafrome of XBRL indices
        num_workers: Number of workers to use
        log_level: Logging level
    Returns: Pandas dataframe of failed records
    """
    df = msg["data"]
    year = msg['year']
    qtr = msg['qtr']
    output_csv_directory = msg["output_csv_directory"]
    output_xml_directory = msg["output_xml_directory"]
    basename = msg['basename']
    num_workers = msg['num_workers']
    log_level = msg['log_level']

    assert df is not None and len(df) > 0, "worker(): invalid dataframe"
    assert year.isdecimal() and re.match(r"^[12][0-9]{3}$", year)
    assert qtr.isdecimal() and re.match(r"^[1-4]$", qtr)
    assert isinstance(num_workers, int) and num_workers > 0
    assert log_level in [10, 20, 30, 40]

    # --------------------------------------------------------------------------------
    # Split dataframe to handle in parallel
    # --------------------------------------------------------------------------------
    assignment = split(tasks=df, num=num_workers)

    # --------------------------------------------------------------------------------
    # Asynchronously invoke tasks
    # --------------------------------------------------------------------------------
    futures = [
        worker.remote({
            "data": task,
            "year": year,
            "qtr": qtr,
            "output_xml_directory": output_xml_directory,
            "log_level": log_level
        })
        for task in assignment
    ]
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
    df = pd.concat(ray.get(waits))

    # --------------------------------------------------------------------------------
    # Save the result dataframe
    # --------------------------------------------------------------------------------
    package = {
        "data": df,
        "directory": output_csv_directory,
        "basename": basename,
        "compression": "gzip",
    }
    save_to_csv(package)

    return df


def load_from_csv(filepath, types=[FS_TYPE_10K, FS_TYPE_10K]):
    """Load each directory listing csv into a pandas dataframe.
    CSV has the format where 'Fillename' is the URL to the XBRL XML.
    |CIK|Company Name|Form Type|Date Filed|Filename|

    Args:
        filepath: path to a XBRL index file
        types: Filing types e.g. 10-K
    Returns:
        pandas dataframe
    """
    logging.info("load_from_csv(): filepath [%s]" % filepath)
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
        # usecols=['CIK', 'Form Type', 'Filename'],
        # parse_dates=['Date Filed'],
    )

    # --------------------------------------------------------------------------------
    # Select filing for the target Form Types e.g. 10-K
    # --------------------------------------------------------------------------------
    listings = listings.loc[listings['Form Type'].isin(types)] if types else listings
    listings.loc[:, 'Form Type'] = listings['Form Type'].astype('category')
    logging.info("load_from_csv(): size of listings [%s]" % len(listings))

    return listings


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------------------------
    # Command line arguments
    # --------------------------------------------------------------------------------
    args = get_command_line_arguments()
    input_csv_directory, output_csv_directory, output_xml_directory, year, qtr, num_workers, log_level = args

    # --------------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------------
    logging.basicConfig(level=log_level)
    # logging.addHandler(logging.StreamHandler())

    # --------------------------------------------------------------------------------
    # XBRL XML logic
    # --------------------------------------------------------------------------------
    try:
        # --------------------------------------------------------------------------------
        # Setup Ray
        # --------------------------------------------------------------------------------
        logging.info("main(): initializing Ray using %s workers..." % num_workers)
        ray.init(num_cpus=num_workers, num_gpus=0, logging_level=log_level)

        # --------------------------------------------------------------------------------
        # Process XBRL listing files
        # --------------------------------------------------------------------------------
        for filepath in list_csv_files(
            input_csv_directory=input_csv_directory, output_csv_directory=output_csv_directory, year=year, qtr=qtr
        ):
            filename = os.path.basename(filepath)
            logging.info("main(): processing the listing csv [%s]..." % filename)

            # --------------------------------------------------------------------------------
            # Load the listing CSV ({YEAR}QTR{QTR}_LIST.gz) into datafame
            # --------------------------------------------------------------------------------
            df = load_from_csv(filepath=filepath, types=[FS_TYPE_10Q, FS_TYPE_10K])

            # --------------------------------------------------------------------------------
            # Year/Quarter of the listing is filed to SEC
            # --------------------------------------------------------------------------------
            match = re.search("^([1-2][0-9]{3})QTR([1-4]).*$", filename, re.IGNORECASE)
            year = match.group(1)
            qtr = match.group(2)
            basename = f"{year}QTR{qtr}"

            # --------------------------------------------------------------------------------
            # Process the single listing file
            # --------------------------------------------------------------------------------
            msg = {
                "data": df,
                "year": year,
                "qtr": qtr,
                "output_csv_directory": output_csv_directory,
                "output_xml_directory": output_xml_directory,
                "basename": basename,
                "num_workers": num_workers,
                "log_level": log_level
            }
            result = director(msg)

            # --------------------------------------------------------------------------------
            # List failed records with 'Filepath' column being None as failed to get XBRL
            # --------------------------------------------------------------------------------
            if any(result['Filepath'].isna()):
                print("*" * 80)
                print("-" * 80)
                print(f"[{len(result['Filepath'].isna())}] failed records:\n")
                print(df)
            else:
                logging.info("main(): all [%s] records processed in %s" % (len(df), filename))
    finally:
        # --------------------------------------------------------------------------------
        # Clean up resource
        # --------------------------------------------------------------------------------
        logging.info("main(): shutting down Ray...")
        ray.shutdown()

