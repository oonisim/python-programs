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
* Investopedia -[Where Can I Find a Company's Annual Report and Its SEC Filings?](https://www.investopedia.com/ask/answers/119.asp)

> If you want to dig deeper and go beyond the slick marketing version of the annual report found on corporate websites, you'll have to search through required filings made to the Securities and Exchange Commission. All publicly-traded companies in the U.S. must file regular financial reports with the SEC. These filings include the annual report (known as the 10-K), quarterly report (10-Q), and a myriad of other forms containing all types of financial data.45

# Quarterly filing indices
* [Accessing EDGAR Data](https://www.sec.gov/os/accessing-edgar-data)

> Using the EDGAR index files  
Indexes to all public filings are available from 1994Q3 through the present and located in the following browsable directories:
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
import random
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
    DIR_CSV_INDEX,
    DIR_CSV_LIST
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
        '-ic', '--input-csv-directory', type=str, required=False, default=DIR_CSV_INDEX,
        help='specify the input csv directory'
    )
    parser.add_argument(
        '-oc', '--output-csv-directory', type=str, required=False, default=DIR_CSV_LIST,
        help='specify the output csv directory'
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

    # Optional
    year = str(args['year']) if args['year'] else None
    qtr = str(args['qtr']) if args['qtr'] else None
    num_workers = args['num_workers'] if args['num_workers'] else NUM_CPUS
    log_level = args['log_level'] if args['log_level'] else DEFAULT_LOG_LEVEL

    return input_csv_directory, output_csv_directory, year, qtr, num_workers, log_level


def csv_absolute_path_to_save(directory, filename):
    """Generate the absolute path to the directory in which to save the csv file
    Args:
        directory: Absolute path to the directory to save the file.
        filename: Filename of the file to save
    """
    if not hasattr(csv_absolute_path_to_save, "mkdired"):
        pathlib.Path(directory).mkdir(mode=0o775, parents=True, exist_ok=True)
        setattr(csv_absolute_path_to_save, "mkdired", True)

    destination = os.path.realpath(f"{directory}{os.sep}{filename}_LIST.gz")
    logging.debug("csv_absolute_path_to_save(): Path to save csv is [%s]" % destination)
    return destination


def list_csv_files(input_csv_directory, output_csv_directory, year=None, qtr=None):
    """List files in the directory
    When year is specified, only matching listing files for the year will be selected.
    When qtr is specified, only match8ng listing files for the quarter will be selected.

    Args:
        input_csv_directory: path to the directory from where to get the file
        output_csv_directory: path to the directory where the listings will be saved
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
        source = pathlib.Path(filepath)
        target = pathlib.Path(csv_absolute_path_to_save(output_csv_directory, os.path.basename(filepath)))

        if source.is_file():
            if target.exists():
                logging.info(
                    "list_csv_files(): directory listing file [%s] already exits. skipping [%s]."
                    % (target.absolute(), filepath)
                )
                return False
            else:
                logging.info("list_csv_files(): adding [%s] to handle" % filepath)
                return True
        else:
            logging.info("[%s] does not exist or not a file. skipping." % filepath)
            return False

    pattern = ""
    pattern += f"{year}" if year else "*"
    pattern += "QTR"
    pattern += f"{qtr}" if qtr else "?"

    logging.info("Listing the files to process in the directory %s ..." % input_csv_directory)
    files = sorted(
        filter(is_file_to_process, glob.glob(input_csv_directory + os.sep + pattern)),
        reverse=True
    )
    if files is None or len(files) == 0:
        logging.info("No files to process in the directory %s" % input_csv_directory)

    return files


def save_to_csv(df, directory, filename):
    """Save the XBRL"""
    destination = csv_absolute_path_to_save(directory, filename)
    logging.info("save_to_csv(): saving XBRL dataframe to [%s]..." % destination)
    df.to_csv(
        destination,
        sep="|",
        compression="gzip",
        header=True,
        index=False
    )


# ================================================================================
# Logic
# ================================================================================
def index_xml_url(filename):
    """
    Generate the URL to the EDGAR directory listing index.xml file in the format
    'https://sec.gov/Archives/edgar/data/{CIK}/{ACCESSION}/index.xml' from the
    directory path to the XBRL TXT file specified in the EDGAR XBRL index CSV.

    | CIK    | Company Name | Form Type          | Date Filed | Filename   |
    |--------|--------------|--------------------|------------|------------|
    |1047127 |AMKOR         |10-K                |2014-02-28  |edgar/data/1047127/0001047127-14-000006.txt|

    Example: From 'edgar/data/1001039/0001193125-10-025949.txt', generate
    'https://sec.gov/Archives/edgar/data/1001039/000119312510025949/index.xml'
    """
    # Verify the filename has the "edgar/data/{CIK}/{ACCESSION_NUMBER}.txt" format
    expected = "edgar/data/{CIK}/{ACCESSION_NUMBER}.txt"
    pattern = r"edgar/data/[0-9]+/[0-9]+[-0-9]+\.txt"
    if re.match(pattern, filename):
        url = "/".join([
            EDGAR_BASE_URL,
            filename.rstrip(".txt").replace('-', ''),
            "index.xml"
        ])
    else:
        logging.error("Expected the format [{}] but got [{}]".format(expected, filename))
        url = None

    return url


def load_from_csv(filepath, types=[FS_TYPE_10K, FS_TYPE_10K]):
    """Load a XBRL master index csv into a pandas dataframe.
    The master index file format must be <YYYY>QTR<N> e.g. 2010QTR1.

    CSV has the format "|CIK|Company Name|Form Type|Date Filed|Filename|"
    'Filename' column is updated with the URL to the directory listing index.xml.

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
    indices = pd.read_csv(
        filepath,
        skip_blank_lines=True,
        header=0,         # The 1st data line after omitting skiprows and blank lines.
        sep='|',
        parse_dates=['Date Filed'],
    )

    # --------------------------------------------------------------------------------
    # Select filing for the target Form Types e.g. 10-K
    # --------------------------------------------------------------------------------
    indices = indices.loc[indices['Form Type'].isin(types)] if types else indices
    indices.loc[:, 'Form Type'] = indices['Form Type'].astype('category')

    # --------------------------------------------------------------------------------
    # Update the 'Filename' column with the URL to index.xml
    # --------------------------------------------------------------------------------
    indices.loc[:, 'Filename'] = indices['Filename'].apply(index_xml_url)
    logging.info("load_from_csv(): size of indices [%s]" % len(indices))

    return indices


def filing_directory_path(index_xml_url: str):
    """
    Get the SEC Filing directory path "/Archives/edgar/data/{CIK}/{ACCESSION}/"
    from the URL to EDGAR directory listing index.xml by removing the base
    "https://[www.]sec.gov" and filename "index.xml".

    Args:
        index_xml_url: URL to the EDGAR directory listing index.xml
    Returns:
        "/Archives/edgar/data/{CIK}/{ACCESSION}/" part in the index_xml_url
    """
    # --------------------------------------------------------------------------------
    # regexp pattern to extract the directory path part (3rd group)
    # --------------------------------------------------------------------------------
    pattern = r"(http|https)://(www\.sec\.gov|sec\.gov)(.*/)index.xml"
    match = re.search(pattern, index_xml_url, re.IGNORECASE)

    # --------------------------------------------------------------------------------
    # Verify the match
    # --------------------------------------------------------------------------------
    assert match and match.group(3) and re.match(r"^/Archives/edgar/data/[0-9]*/[0-9]*/", match.group(3)), \
        f"regexp [{pattern}] found No matching directory path in url {index_xml_url}.\n{match}"

    path = match.group(3)
    logging.debug("Filing directory path is [%s]" % path)
    return path


def find_xbrl_url_from_html_xml(source, directory):
    """Find XBRL XML with suffix _htm.xml in the Filing directory
    Args:
        source: Content of filing directory listing index.xml (as BS4)
        directory: Filing directory path "/Archives/edgar/data/{CIK}/{ACCESSION}/"
        logger: worker logger
    Returns:
        URL to the XBRL XML if found, else None
    """
    assert isinstance(source, bs4.BeautifulSoup)
    path_to_xbrl = source.find('href', string=re.compile(".*_htm\.xml"))
    if path_to_xbrl:
        url = "https://sec.gov" + path_to_xbrl.string.strip()
        logging.info("XBRL XML [%s] identified" % url)
        return url
    else:
        logging.warning("No XBRL XML pattern [%s.*_htm,.xml] identified." % directory)
        return None


def find_xbrl_url_from_xsd(source, directory):
    """Find XBRL XML having similar name with .xsd file in the Filing directory.
    XBRL XML requires XSD file and the XBRL is likely to have the same file name
    of the XSD file, e.g. <filename>.xml and <filename>.xsd.

    <filename> does not always 100% match, e.g. asc-20191223.xsd and asc-20190924.
    Hence, use the LEVENSHTEIN DISTANCE to fuzzy match.

    Args:
        source: Content of filing directory listing index.xml (as BS4)
        directory: Filing directory path "/Archives/edgar/data/{CIK}/{ACCESSION}/"
        logger: worker logger
    Returns:
        URL to XBRL XML if found, else None
    """
    assert isinstance(source, bs4.BeautifulSoup)
    MAX_LEVENSHTEIN_DISTANCE = 5

    path_to_xsd = source.find('href', string=re.compile(re.escape(directory) + ".*\.xsd"))
    if path_to_xsd:
        # Extract filename from "/Archives/edgar/data/{CIK}/{ACCESSION}/<filename>.xsd".        
        pattern = re.escape(directory) + r"(.*)\.xsd"
        path_to_xsd = path_to_xsd.string.strip()
        match = re.search(pattern, path_to_xsd, re.IGNORECASE)
        assert match and match.group(1), f"No filename match for with {pattern}"

        # Filename of the XSD
        filename = match.group(1)

        # Iterate over all .xml files and find the distance from the XSD filename.
        distance = 999
        candidate = None
        for href in source.find_all('href', string=re.compile(re.escape(directory) + ".*\.xml")):
            pattern = re.escape(directory) + r"(.*)\.xml"
            match = re.search(pattern, href.string.strip(), re.IGNORECASE)
            assert match and match.group(1), f"[{href}] has no .xml with {pattern}"

            potential = match.group(1)
            new_distance = levenshtein.distance(filename, potential)
            if new_distance < distance:
                distance = new_distance
                candidate = potential
                logging.debug(
                    "Candidate [%s] is picked with the distance from [%s] is [%s]."
                    % (candidate, filename, distance)
                )

        # Accept within N-distance away from the XSD filename.
        if distance < MAX_LEVENSHTEIN_DISTANCE:
            path_to_xml = directory + candidate.strip() + ".xml"
            url = "https://sec.gov" + path_to_xml
            logging.debug(
                "Selected the candidate [%s] of distance [%s]. \nURL to XBRL is [%s]"
                % (candidate, distance, url)
            )
            logging.info("XBRL XML [%s] identified" % url)
            return url
        else:
            logging.warning(
                "No XBRL XML identified from the XSD file [%s%s]."
                % (directory, filename + ".xsd")
            )
            return None
    else:
        logging.error("No XBRL identified from XSD [%s]." % path_to_xsd)
        return None


def find_xbrl_url_from_auxiliary_regexp(source, directory):
    """Find XBRL XML with regexp pattern.
    XBRL XML file most likely has the format if <str>-<YEAR><MONTH><DATE>.xml,
    but should NOT be Rnn.xml e.g. R1.xml or R10.xml.

    Args:
        source: Content of filing directory listing index.xml (as BS4)
        directory: Filing directory path "/Archives/edgar/data/{CIK}/{ACCESSION}/"
        logger: worker logger
    Returns:
        URL to the XBRL XML if found, else None
    """
    assert isinstance(source, bs4.BeautifulSoup)
    regexp = re.escape(directory) + r"[^R][a-zA-Z_-]*[0-9][0-9][0-9][0-9][0-9].*\.xml"
    logging.debug("Look for XBRL XML with the regexp [%s]." % regexp)

    path_to_xbrl = source.find('href', string=re.compile(regexp))
    if path_to_xbrl:
        url = "https://sec.gov" + path_to_xbrl.string.strip()
        logging.info("XBRL XML [%s] identified" % url)
        return url
    else:
        logging.warning(
            "No XBRL XML identified with the regexp [%s] in [%s]." % (regexp, directory)
        )
        return None


def xbrl_url(index_xml_url: str):
    """Generate the URL to the XBML file in the filing directory 
    Args:
        index_xml_url: 
            URL to the EDGAR directory listing index.xml file whose format is e.g.:
            "https://sec.gov/Archives/edgar/data/62996/000095012310013437/index.xml"
    Returns:
        URL to the XBRL file in the filing directory, or None
    """
    # --------------------------------------------------------------------------------
    # URL to directory listing index.xml
    # --------------------------------------------------------------------------------
    index_xml_url = index_xml_url.strip()
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

            logging.info("xbrl_url(): getting filing directory index [%s]..." % index_xml_url)
            content = http_get_content(index_xml_url, EDGAR_HTTP_HEADERS)
            break
        except RuntimeError as e:
            max_retries_allowed -= 1
            if max_retries_allowed > 0:
                logging.error("xbrl_url(): failed to get [%s]. retrying..." % index_xml_url)
                time.sleep(random.randint(30, 90))
            else:
                content = None
                logging.error("xbrl_url(): failed to get [%s]. skipping..." % index_xml_url)
                break

    if content is not None:
        logging.info(f"identifying XBRL URL for [%s]..." % index_xml_url)
        # --------------------------------------------------------------------------------
        # "/Archives/edgar/data/{CIK}/{ACCESSION}/" part of the EDGAR filing URL
        # --------------------------------------------------------------------------------
        directory = filing_directory_path(index_xml_url)

        # --------------------------------------------------------------------------------
        # Look for the XBRL XML file in the index.xml.
        # 1. _htm.xml file
        # 2. <filename>.xml where "filename" is from <filename>.xsd.
        # 3. <filename>.xml where "filename" is not RNN.xml e.g. R10.xml.
        # --------------------------------------------------------------------------------
        source = BeautifulSoup(content, 'html.parser')
        url = find_xbrl_url_from_html_xml(source=source, directory=directory)
        if url:
            logging.info(f"XBRL URL identified [%s]" % url)
            return url

        url = find_xbrl_url_from_xsd(source, directory=directory)
        if url:
            logging.info(f"XBRL URL identified [%s]" % url)
            return url

        url = find_xbrl_url_from_auxiliary_regexp(source=source, directory=directory)
        if url:
            logging.info(f"XBRL URL identified [%s]" % url)
            return url

        logging.error("No XBRL identified in the listing [%s]" % index_xml_url)
        return None
    else:
        return None


def test_xbrl_url():
    # --------------------------------------------------------------------------------
    # Test the sample filling which has irregular XBRL fliename pattern.
    # --------------------------------------------------------------------------------
    SAMPLE_CIK = "62996"
    SAMPLE_ACC = "000095012310013437"
    SAMPLE_DIRECTORY_LISTING = "/".join([
        "https://sec.gov/Archives/edgar/data",
        SAMPLE_CIK,
        SAMPLE_ACC,
        "index.xml",
    ])
    print("XBRL XML URL For {}\n{}".format(
        SAMPLE_DIRECTORY_LISTING,
        xbrl_url(SAMPLE_DIRECTORY_LISTING)
    ))


@ray.remote(num_returns=1)
def worker(df, log_level: int):
    """GET XBRL XML and save to a file.
    Args:
        df: Pandas dataframe |CIK|Company Name|Form Type|Date Filed|Filename|
            where 'Filename' is the URL to index.xml in the filing directory.
        log_level: Logging level to use in the worker.
    Returns: Pandas dataframe where "Filename" column is updated with XBRL XML URL.
    """
    assert df is not None and len(df) > 0, "worker(): invalid dataframe"
    # --------------------------------------------------------------------------------
    #  Logging setup for Ray as in https://docs.ray.io/en/master/ray-logging.html.
    #  In Ray, all of the tasks and actors are executed remotely in the worker processes.
    #  Since Python logger module creates a singleton logger per process, loggers should
    #  be configured on per task/actor basis.
    # --------------------------------------------------------------------------------
    logging.basicConfig(level=log_level)

    assert len(df) > 0
    logging.info("worker(): task size is %s" % len(df))
    df.loc[:, 'Filename'] = df['Filename'].apply(xbrl_url)

    return df


def director(df, num_workers: int, log_level: int):
    """Director to generate the URL to XBRL XML files in the SEC filing directory.
    Directory listing index.xml provides the list of files in the filing directory.
    Identify the XBRL XML in the directory and generate the URL to the XML.

    Args:
        df: pandas datafrome of XBRL indices
        num_workers: Number of workers to use

    Returns:
        Pandas dataframe |CIK|Company Name|Form Type|Date Filed|Filename|
        where 'Filename' column set to the URL to XBRL XML.
    """
    # --------------------------------------------------------------------------------
    # Split dataframe to handle in parallel
    # --------------------------------------------------------------------------------
    assignment = split(tasks=df, num=num_workers)

    # --------------------------------------------------------------------------------
    # Asynchronously invoke tasks
    # --------------------------------------------------------------------------------
    futures = [worker.remote(task, log_level) for task in assignment]
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
    result = pd.concat(ray.get(waits))
    result.sort_index(inplace=True)

    return result


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------------------------
    # Command line arguments
    # --------------------------------------------------------------------------------
    args = get_command_line_arguments()
    input_csv_directory, output_csv_directory, year, qtr, num_workers, log_level = args

    # --------------------------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------------------------
    logging.basicConfig(level=log_level)

    # --------------------------------------------------------------------------------
    # XBRL XML logic
    # --------------------------------------------------------------------------------
    try:
        # --------------------------------------------------------------------------------
        # Setup Ray
        # --------------------------------------------------------------------------------
        logging.info("main(): initializing Ray using %s workers..." % num_workers)
        ray.init(num_cpus=num_workers, num_gpus=0, logging_level=log_level)

        for filepath in list_csv_files(
            input_csv_directory=input_csv_directory, output_csv_directory=output_csv_directory, year=year, qtr=qtr
        ):
            filename = os.path.basename(filepath)
            logging.info("main(): processing the listing [%s]..." % filename)

            result = director(
                df=load_from_csv(filepath=filepath, types=[FS_TYPE_10Q, FS_TYPE_10K]),
                num_workers=num_workers,
                log_level=log_level
            )
            failure_df = result.loc[result['Filename'].isna()]
            success_df = result.loc[result['Filename'].notna()]
            if len(failure_df) > 0:
                logging.error("main(): %s records failed." % len(failure_df))
                logging.error("main(): Failed records:\n%s." % failure_df)
            else:
                logging.info("main(): processed %s records for %s" % (len(result), filename))

            save_to_csv(success_df, directory=output_csv_directory, filename=filename)
    finally:
        # --------------------------------------------------------------------------------
        # Clean up resource
        # --------------------------------------------------------------------------------
        logging.info("main(): shutting down Ray...")
        ray.shutdown()

