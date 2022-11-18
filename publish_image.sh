PROJECT_ID="stone-bounty-367715"
REGION="us-west1"
REPOSITORY="ml-ete-p1"
IMAGE="ml-ete-p1"
VERSION="0.0.1"

gcloud beta artifacts repositories create $REPOSITORY \
 --repository-format=docker \
 --location=$REGION
 
 
# Configure Docker
gcloud auth configure-docker $REGION-docker.pkg.dev
 
 # Push
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE:$VERSION
