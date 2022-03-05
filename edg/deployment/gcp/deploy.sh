#!/usr/bin/env bash
set -eu

echo "--------------------------------------------------------------------------------"
echo "Setting up GCP project"
echo "--------------------------------------------------------------------------------"
export REGION=us-central1
export PROJECT_ID=$(gcloud config get-value project)
export PROJECT_NUMBER="$(gcloud projects describe ${PROJECT_ID} --format='get(projectNumber)')"
export GKE_CLUSTER_NAME="edg"
export APP_NAME="diabetes-prediction"

echo "--------------------------------------------------------------------------------"
echo "Setting up API"
echo "--------------------------------------------------------------------------------"
gcloud services enable container.googleapis.com \
    cloudbuild.googleapis.com \
    sourcerepo.googleapis.com \
    containeranalysis.googleapis.com

echo "--------------------------------------------------------------------------------"
echo "Setting up IAM policy to access GKE from Cloud Build"
echo "--------------------------------------------------------------------------------"
gcloud projects add-iam-policy-binding ${PROJECT_NUMBER} \
    --member=serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com \
    --role=roles/container.developer

echo "--------------------------------------------------------------------------------"
echo "Setting up Artifact registry"
echo "--------------------------------------------------------------------------------"
gcloud artifacts repositories create ${APP_NAME} \
  --repository-format=docker \
  --location=${REGION}

echo "--------------------------------------------------------------------------------"
echo "Setting up GKE"
echo "--------------------------------------------------------------------------------"
gcloud container clusters create ${GKE_CLUSTER_NAME} \
    --num-nodes 1 --region ${REGION}

gcloud container clusters get-credentials ${GKE_CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}
