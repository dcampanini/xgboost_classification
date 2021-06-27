echo -e "\nDefiniendo variables de entorno\n"
PROJECT_ID=teco-mvp-aiops
BUCKET_ID=teco_ai
JOB_NAME=teco_training_$(date +"%Y%m%d_%H:%M:%S")
JOB_DIR=gs://$BUCKET_ID/output_train_teco
TRAINING_PACKAGE_PATH="/home/diego_campanini/census_training"
MAIN_TRAINER_MODULE=census_training.aitrain
REGION=us-central1
RUNTIME_VERSION=1.15
PYTHON_VERSION=2.7
SCALE_TIER=BASIC
echo -e "PROJECT_ID=$PROJECT_ID" "\nBUCKET_ID=$BUCKET_ID" " \nJOB_NAME=$JOB_NAME"
echo -e "JOB_DIR=$JOB_DIR" "TRAINING_PACKAGE_PATH=$TRAINING_PACKAGE_PATH"
echo -e "MAIN_TRAINER_MODULE=$MAIN_TRAINER_MODULE" "\nREGION=$REGION"
echo -e "RUNTIME_VERSION=$RUNTIME_VERSION" "\nPYTHON_VERSION=$PYTHON_VERSION"
echo -e "SCALE_TIER=$SCALE_TIER"

echo -e "\nSubmit the training with ai platform"

gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region $REGION \
  --runtime-version=$RUNTIME_VERSION \
  --python-version=$PYTHON_VERSION \
  --scale-tier $SCALE_TIER