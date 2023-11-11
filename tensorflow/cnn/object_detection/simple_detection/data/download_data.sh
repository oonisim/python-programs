#!/usr/bin/env bash
# https://www.kaggle.com/datasets/techzizou/labeled-mask-dataset-yolo-darknet/data
# PATH_TO_DATA_DIR=$1
set -ue

export KAGGLE_USERNAME=$(cat ~/.kaggle/kaggle.json | jq -r '.username')
export KAGGLE_KEY=$(cat ~/.kaggle/kaggle.json | jq -r '.key')
# kaggle datasets download -d techzizou/labeled-mask-dataset-yolo-darknet -p .
unzip -o *.zip
# find ./obj -name "*[ ()]*" -execdir rename 's/[ ()]//g; s/\(//g' "{}" \;
find ./obj -name "*[ ()]*" -execdir rename 's/[ ()]//g' "{}" \;