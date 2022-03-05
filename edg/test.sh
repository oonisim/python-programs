#!/usr/bin/env bash
set -e

docker build -t edg -f src/serving/Dockerfile .
docker run --publish 8080:8080 edg