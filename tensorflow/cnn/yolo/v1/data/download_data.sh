#!/usr/bin/env bash
set -ue


export KAGGLE_USERNAME=$(cat ~/.kaggle/kaggle.json | jq -r '.username')
export KAGGLE_KEY=$(cat ~/.kaggle/kaggle.json | jq -r '.key')
kaggle datasets download -d aladdinpersson/pascalvoc-yolo -p $PATH_TO_DATA_DIR