#!/bin/bash     
PROJECT_ID="stone-bounty-367715"
REGION="us-west1"
REPOSITORY="ml-ete-p1"
IMAGE="ml-ete-p1"
VERSION="0.0.1"

docker build --tag=$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE:$VERSION .