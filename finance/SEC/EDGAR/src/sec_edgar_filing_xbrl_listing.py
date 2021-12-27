#!/usr/bin/env python
"""
# Load SEC EDGAR Quarterly (10-Q) XBRL

## Objective
Navigate through the EDGAR XBRL listings to generate the URLs to XBRL XML files.

## XBRL Listing File
Each row in https://sec.gov/Archives/edgar/full-index/${YEAR}/${QTR}/xbrl.gz
tells where the TXT file of 10-Q or 10-K for a CIK.

| CIK    | Company Name | Form Type          | Date Filed | Filename   |                                           
|--------|--------------|--------------------|------------|------------|
|1047127|AMKOR TECHNOLOGY INC|10-K|2014-02-28|edgar/data/1047127/0001047127-14-000006.txt|

## TXT File
TXT file has multiple <DOCUMENT> tags, each of which corresponds to a document
in the SEC filing for (CIK, YEAR, QTR).

## XBRL File
A filing has a XBRL XML file which has the Item 8 F/S data. Instead of parsing
the TXT file, directly part the XML file. However, the format fo the filename
is not consistent, e.g. <name>_htm.xml, <name>.xml. Hence needs to identify the
XBRL XML file in each listing.

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

# --------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------
import argparse
import os
import logging
import glob
import re
import json
import time
import requests

import ray
from bs4 import BeautifulSoup
import pandas as pd
import Levenshtein as levenshtein 

pd.set_option('display.max_colwidth', None)
logging.basicConfig(level=logging.ERROR)
Logger = logging.getLogger(__name__)

NUM_CPUS = 8
FS_TYPE_10K = "10-K"
FS_TYPE_10Q = "10-Q"
EDGAR_BASE_URL = "https://sec.gov/Archives"
EDGAR_HTTP_HEADERS = {"User-Agent": "Company Name myname@company.com"}
DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = f"{DIR}/../data/listings/XBRL"


def list_files(directory, year=None, qtr=None):
    """List files in the directory
    When year is specified, only matching listing files for the year will be selected.
    When qtr is specified, only match8ng listing files for the quarter will be selected.

    Args:
        directory: path to the directory from where to get the file
        year: Year to select
        qtr: Qauater to select
    Returns: List of flies
    """
    assert os.path.isdir(directory), f"Not a directory or does not exist: {directory}"
    assert (re.match(r"[1-2][0-9][0-9][0-9]", year) if year else True), f"Invalid year {year}"
    assert (re.match(r"[1-4]", qtr) if qtr else True), f"Invalid quarter {qtr}"

    pattern = ""
    pattern += f"{year}" if year else "*"
    pattern += "QTR"
    pattern += f"{qtr}" if qtr else "?"

    # Sort the files alphabetically
    return sorted(filter(os.path.isfile, glob.glob(directory + os.sep + pattern)))


def index_xml_url(filename):
    """Generate the EDGAR directory listing index.xml URL.
    https://www.sec.gov/Archives/edgar/data/{CIK}/{ACCESSION}/index.xml"
    """
    url = "/".join([
        EDGAR_BASE_URL,
        filename.rstrip(".txt").replace('-', ''),
        "index.xml"
    ])
    return url


def edgar_xbrl_listing_file_datafarme(directory, year=None, qtr=None, types=[FS_TYPE_10K, FS_TYPE_10K]):
    """
    Generate a pandas dataframe for each XBRL listing file.
    'Filename' column is updated with the URL for the EDGAR directory listing index.xml.

    The EDGAR listing file format must be <YYYY>QTR<N> e.g. 2010QTR1.
    When year is specified, only matching listing files for the year are processed.
    When qtr is specified, only match8ng listing files for the quarter are processed.
    
    Args:
        directory: directory where XBRL files are located.
        year: Filing year (YYYY)
        qtr: Filing quarter either 1, 2, 3, 4
        types: Filing types e.g. 10-K
    Returns:
        pandas df    
    """
    files = list_files(directory=directory, year=year, qtr=qtr)
    for filepath in files:
        filename = os.path.basename(filepath)
        if os.path.isfile(f"{filepath}_XBRL.gz"):
            Logger.debug(f"{filepath} exists. Skip...")
            continue
            
        df = pd.read_csv(
            filepath,
            skip_blank_lines=True,
            header=0,         # The 1st data line after omitting skiprows and blank lines.
            sep='|',
            parse_dates=['Date Filed'],
        )
        
        # Select rows for target filing types
        df = df.loc[df['Form Type'].isin(types)] if types else df
        
        # Set the index.xml URL as the filename
        df['Filename'] = df['Filename'].apply(index_xml_url)

        yield filename, df


def xbrl_url(index_xml_url: str):
    """Generate the URL to the XBML file in the filing directory 
    Args:
        index_xml_url: 
            URL to the EDGAR directory listing index.xml file whose format is e.g.:
            "https://sec.gov/Archives/edgar/data/62996/000095012310013437/index.xml"
    Returns:
        URL to the XBRL file in the filing directory.
    """
    MAX_LEVENSHTEIN_DISTANCE = 5
    
    index_xml_url = index_xml_url.strip()
    Logger.debug(f"Identifying XBRL URLs for the listing [%s]" % index_xml_url)

    # --------------------------------------------------------------------------------
    # Get the filing directory path "/Archives/edgar/data/{CIK}/{ACCESSION}/" from
    # index_xml_url by removing "https://[www.]sec.gov" and "index.xml".
    # --------------------------------------------------------------------------------
    pattern = r"(http|https)://(www\.sec\.gov|sec\.gov)(.*/)index.xml"
    match = re.search(pattern, index_xml_url, re.IGNORECASE)
    assert match and re.match(r"^/Archives/edgar/data/[0-9]*/[0-9]*/", match.group(3)), \
        f"No matching path found by regexp [{pattern}], but got {match}"

    directory = match.group(3)
    Logger.debug("Filing directory path is [%s]" % directory)
    
    # --------------------------------------------------------------------------------
    # GET the index.xml
    # --------------------------------------------------------------------------------
    response = requests.get(index_xml_url, headers=EDGAR_HTTP_HEADERS)
    if response.status_code == 200:
        content = response.content.decode("utf-8") 
    else:
        Logger.error("%s failed with %s" % (index_xml_url, response.status_code))
        assert False, f"{index_xml_url} failed with status {response.status_code}"
    
    # --------------------------------------------------------------------------------
    # Look for the XBRL XML file in the index.xml.
    # 1. _htm.xml file
    # 2. <filename>.xml where "filename" is from <filename>.xsd.
    # 3. <filename>.xml where "filename" is not RNN.xml e.g. R10.xml.
    # --------------------------------------------------------------------------------
    # 1. Look for _htm.xml.
    index = BeautifulSoup(content, 'html.parser')
    # print(index.prettify())
    
    path_to_xbrl = index.find('href', string=re.compile(".*_htm\.xml"))
    if path_to_xbrl:
        url = "https://sec.gov" + path_to_xbrl.string.strip()
        Logger.debug("URL to XBRL is [%s]" % url)
        return url
    else:
        Logger.warning(f"No XBRL with the .*_htm,.xml pattern in the listing {index_xml_url}")

    # 2. Look for XML file for the corresponding XSD.
    path_to_xsd = index.find('href', string=re.compile(re.escape(directory) + ".*\.xsd"))
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
        for href in index.find_all('href', string=re.compile(re.escape(directory) + ".*\.xml")):
            pattern = re.escape(directory) + r"(.*)\.xml"
            match = re.search(pattern, href.string.strip(), re.IGNORECASE)
            assert match and match.group(1), f"[{href}] has no .xml with {pattern}"

            potential = match.group(1)
            new_distance = levenshtein.distance(filename, potential)
            if new_distance < distance:
                distance = new_distance
                candidate = potential
                Logger.debug(
                    "Candidate [%s] is picked with the distance from [%s] is [%s]." 
                    % (candidate, filename, distance) 
                )
        
        # Accept within N-distance away from the XSD filename.
        if distance < MAX_LEVENSHTEIN_DISTANCE:  
            path_to_xml = directory + candidate + ".xml"
            url = "https://sec.gov" + path_to_xml
            Logger.debug(
                "Selected the candidate [%s] of distance [%s]. \nURL to XBRL is [%s]" 
                % (candidate, distance, url)
            )
            return url
        else:
            Logger.warning(
                "No corresponding XBRL found for the XSD file [%s]." % (filename + ".xsd")
            )
    else:
        Logger.error("No XSD file found in the listing [%s]." % index_xml_url)
    
    # 3. Look for XML with href="/Archives/edgar/data/{CIK}/{ACCESSION}/[^R0-9]*\.xml"
    # Regexp to match the XBRL XML file which is NOT Rnn.xml e.g. R1.xml or R10.xml. 
    # Most likely it has the format if <str>-<YEAR><MONTH><DATE>.xml.
    regexp = re.escape(directory) + r"[^R][a-zA-Z_-]*[0-9][0-9][0-9][0-9][0-9].*\.xml"
    Logger.debug("Look for XBRL XML with the regexp [%s]." % regexp)

    path_to_xbrl = index.find('href', string=re.compile(regexp))
    if path_to_xbrl:
        url = "https://sec.gov" + path_to_xbrl.string.strip()
        Logger.debug("Identified the XBRL URL [%s]." % url)
        return url
    else:
        Logger.warning("No XBRL filename matched with the regexp [%s]." % regexp)

    Logger.error("No XBRL identified in the listing [%s]" % index_xml_url)
    assert False, "No XBRL found. Check [%s] to identify the XBRL." % index_xml_url

    # time.sleep(1)


def generate_xbrl_xml_urls(index_xml_urls):
    """Generate the XBRL XML URL for the SEC filing directory listing index.xml
    index.xml provides the list of files in the filing directory.
    Identify the XBRL XML in the directory and generate the URL to the XML.
    """
    # TODO: Parallel execution
    raise NotImplementedError("To be implemented")


def save_to_csv(df, directory, filename):
    """Save"""
    assert os.path.isdir(directory), f"Not a directory or does not exist: {directory}"
    df.to_csv(f"{directory}/{filename}_XBRL.gz", sep="|", compression="gzip", header=True, index=False)


def test():
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


def get_command_line_arguments():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(description='argparse test program')
    parser.add_argument('-y', '--year', type=int, required=False, help='specify the target year')
    parser.add_argument('-q', '--qtr', type=int, choices=[1, 2, 3, 4], required=False, help='specify the target quarter')
    parser.add_argument('-n', '--num-cpus', type=int, required=False, help='specify the number of cpus to use')
    parser.add_argument(
        '-l', '--log_level', type=int, choices=[10, 20, 30, 40], required=False,
        help='specify the logging level (10 for INFO)',
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = get_command_line_arguments()
    year = str(args['year'])
    qtr = str(args['qtr'])
    num_cpus = args['num-cpus'] if args['num-cpus'] else NUM_CPUS

    try:
        ray.init(num_cpus=NUM_CPUS, num_gpus=0, logging_level=logging.ERROR)

        for filename, df in edgar_xbrl_listing_file_datafarme(directory=DATA_DIR, year=year, qtr=qtr, types=[FS_TYPE_10Q]):
            print(f"Processing the listing [{filename}]...")

            # df['XBRL'] = df['Filename'].apply(xbrl_url)
            df['XBRL'] = generate_xbrl_xml_urls(df['Filename'].tolist())

            save_to_csv(df, directory=DATA_DIR, filename=filename)
    finally:
        ray.shutdown()

