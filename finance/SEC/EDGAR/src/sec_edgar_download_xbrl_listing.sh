#!/usr/bin/env bash
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