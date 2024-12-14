#!/bin/bash
set -e

# open file limit
ulimit -n 65536 1048576

export PATH=/usr/local/cuda/bin:$PATH

source .venv/bin/activate

PROJECT_DIR="/model/kodama/tuning_competition2025" # FIXME: Change this to your project directory.
export TMPDIR=${PROJECT_DIR}/tmp

# distributed settings
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# GPU settings
export NUM_GPU_PER_NODE=8
NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS=$((NUM_NODES * NUM_GPU_PER_NODE))
echo "NUM_NODES=${NUM_NODES}, NUM_GPU_PER_NODE=${NUM_GPU_PER_NODE}, NUM_GPUS=${NUM_GPUS}"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# for debugging
#export LOGLEVEL=INFO
#export NCCL_DEBUG=WARN
#export NCCL_DEBUG_SUBSYS=WARN

NAME="sft-"$(tr -dc 0-9A-Za-z < /dev/urandom | fold -w 10 | head -1)
MODEL=llm-jp-3-13b
MODEL_PATH=${PROJECT_DIR}/checkpoints/hf-to-nemo/llm-jp--llm-jp-3-13b  # FIXME: Change this to your model path.

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_TC=106 \
  -bind-to none -map-by slot \
  -x PATH \
  -x CUDA_LAUNCH_BLOCKING \
  python train.py \
  trainer.num_nodes=${NUM_NODES} \
  use_mpi=True \
  name=${NAME} \
  model=${MODEL} \
  model.restore_from_path=${MODEL_PATH}
