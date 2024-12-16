#!/bin/bash
set -e

# open file limit
ulimit -n 65536 1048576

source .venv/bin/activate

PROJECT_DIR="/model/kodama/tuning_competition2025" # FIXME: Change this to your project directory.
export TMPDIR=${PROJECT_DIR}/tmp

MODEL_NAME=$1
INPUT_NAME_OR_PATH=${PROJECT_DIR}/result/${MODEL_NAME}
INPUT_HF_PATH="llm-jp/llm-jp-3-13b" # FIXME: Set the huggingface checkpoint path or model name.
OUTPUT_PATH=${PROJECT_DIR}/checkpoints/nemo-to-hf
MODEL_ID=$(basename ${INPUT_NAME_OR_PATH}) # The lowest directory name in INPUT_NAME_OR_PATH is used as the model id. You can specify the model id by overwriting this variable.


if [ -d "${INPUT_NAME_OR_PATH}/checkpoints" ]; then
  INPUT_NAME_OR_PATH="${INPUT_NAME_OR_PATH}/checkpoints"
fi

if [ ! -f "${INPUT_NAME_OR_PATH}/model_config.yaml" ]; then
  echo "model_config.yaml not found in ${INPUT_NAME_OR_PATH}"
  exit 1
fi

if [ ! -d "${INPUT_NAME_OR_PATH}/model_weights" ]; then
  ln -s $(ls -d ${INPUT_NAME_OR_PATH}/step=*) ${INPUT_NAME_OR_PATH}/model_weights
fi


# run
python scripts/ckpt/convert_llama_nemo_to_hf.py \
  --input-name-or-path ${INPUT_NAME_OR_PATH} \
  --input-hf-path ${INPUT_HF_PATH} \
  --output-path ${OUTPUT_PATH} \
  --model-id ${MODEL_ID} \
  --cpu-only \
  --n-jobs 96
