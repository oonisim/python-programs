#!/usr/bin/env bash

echo "--------------------------------------------------------------------------------"
echo "Setting up GCP project"
echo "--------------------------------------------------------------------------------"
export REGION=us-central1
export PROJECT_ID=$(gcloud config get-value project)
export PROJECT_NUMBER="$(gcloud projects describe ${PROJECT_ID} --format='get(projectNumber)')"
export GKE_CLUSTER_NAME="edg"
export APP_NAME="diabetes-prediction"
export COMMIT_ID="$(git rev-parse --short=7 HEAD)"

echo "--------------------------------------------------------------------------------"
echo "Building docker image and pushing it to the artifact registry"
echo "--------------------------------------------------------------------------------"
gcloud builds submit \
  --config cloudbuild.yaml
#  --tag="us-central1-docker.pkg.dev/${PROJECT_ID}/${APP_NAME}/${APP_NAME}-api:${COMMIT_ID}" .
