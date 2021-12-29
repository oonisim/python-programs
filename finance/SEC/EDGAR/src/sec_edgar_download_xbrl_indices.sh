#!/usr/bin/env bash
#--------------------------------------------------------------------------------
# Download EDGAR XBRL master index files.
# https://www.sec.gov/Archives/edgar/full-index stores the master index files.
#
# [xbrl.gz] master index file for  TXT file in XBRL version
# |CIK    |Company Name|Form Type|Filing Date|TXT Path                                   |
# |1002047|NetApp, Inc.|10-Q     |2020-02-18 |edgar/data/1002047/0001564590-20-005025.txt|
#
# [master.gz] master index file for  TXT file in HTML version
# |CIK   |Company Name|Form Type|Filing Date|TXT Path                                   |
# |000180|SANDISK CORP|10-Q      |2006-08-10|edgar/data/1000180/0000950134-06-015727.txt|
#
# Previously, XBRL was not introduced or not mandatory, hence the documents are
# in HTML format and the TXT is the all-in-one including all the filing data
# sectioned by <DOCUMENT>.
#--------------------------------------------------------------------------------
set -e
DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${DIR}

DATA_DIR=$(realpath ../data/listings/XBRL)
mkdir -p ${DATA_DIR}


SEC_EDGAR_LISTING_BASE_URL="https://www.sec.gov/Archives/edgar/full-index"
YEAR=2010
while [ ${YEAR} -le 2021 ]
do
    QTR=1
    while [ ${QTR} -le 4 ]
    do
      echo "${YEAR} QTR ${QTR}..."
        curl --silent -C - \
        --header 'User-Agent:Company Name myname@company.com' \
        --output "${DATA_DIR}/${YEAR}QTR${QTR}.gz" \
        "${SEC_EDGAR_LISTING_BASE_URL}/${YEAR}/QTR{$QTR}/xbrl.gz"

        sleep 1
        QTR=$((${QTR}+1))
    done

    YEAR=$((${YEAR}+1))
done


cd ${DATA_DIR}
for gz in $(ls *.gz)
do
  gunzip -f ${gz}
  # Remove non-csv lines
  sed -i -e '1,/^[ \t]*$/d; /^[- \t]*$/d' $(basename ${gz} .gz)
done