#!/bin/bash

MODEL_NAME="qwen25-72b"
WORKERS=1
BASE_URL="http://default-base-url" 
API_KEY="default-api-key"           

if [ $# -ge 1 ]; then
    MODEL_NAME=$1
fi
if [ $# -ge 2 ]; then
    WORKERS=$2
fi
if [ $# -ge 3 ]; then
    BASE_URL=$3
fi
if [ $# -ge 4 ]; then
    API_KEY=$4
fi

echo "Running evaluation for model: ${MODEL_NAME} with workers: ${WORKERS}"
echo "API Base URL: ${BASE_URL}"
echo "API Key: ${API_KEY}"
echo "Start time: $(date)"


echo "=== Running BERT for Query Type classification tasks ==="

cd ../TimeFore/RSR_BERT/
python main.py --task='test' --epochs=5 --lr=1e-6 --batch_size=16


echo "=== Running Table_Search tasks ==="
cd ../HDR_Table_retrival

python main.py --task='prediction' --model_name="${MODEL_NAME}" --base_url="${BASE_URL}" --api_key="${API_KEY}" --workers="${WORKERS}"
python main.py --task='eval' --model_name="${MODEL_NAME}"


echo "=== Running SQL_Generate tasks ==="
cd ../HDR_SQL_Generate/
python main.py --task='prediction' --model_name="${MODEL_NAME}" --base_url="${BASE_URL}" --api_key="${API_KEY}" --workers="${WORKERS}"
python main.py --task='transform' --model_name="${MODEL_NAME}"
python main.py --task='eval' --model_name="${MODEL_NAME}"


echo "=== Running TimesNet tasks ==="
cd ../TSF_TimesNet
python prediction.py --task='application' --model_name="${MODEL_NAME}" --base_url="${BASE_URL}" --api_key="${API_KEY}" --input_file="../HDR_SQL_Generate/results/${MODEL_NAME} SQL Prediction Results.json"


echo "=== Running Final_Prediction tasks ==="
cd ../RSR_Final_Prediction
python main.py --task='prediction' --model_name="${MODEL_NAME}" --base_url="${BASE_URL}" --api_key="${API_KEY}" --workers="${WORKERS}"
python main.py --task='eval' --model_name="${MODEL_NAME}"


echo "=== All evaluations completed for model: ${MODEL_NAME} ==="
echo "End time: $(date)"
