#!/usr/bin/env bash
set -ue

DIR=$(realpath $(dirname $0))
cd ${DIR}

URL="https://qdmzi4ohlg.execute-api.ap-southeast-2.amazonaws.com/prod"
files=(
  "august"
  "september"
  "october"
  "november"
  "schedules"
)

for f in ${files[@]}
do
  echo "downloading ${URL}/${f}"
  rm -f ${f} "${f}.json"

  wget --quiet --directory-prefix="${DIR}" "${URL}/${f}"
  jq . < ${f} > "${f}.json"

  rm -f ${f}
done
