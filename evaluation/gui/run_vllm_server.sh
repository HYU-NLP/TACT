#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_ENFORCE_EAGER=true
# export VLLM_DISABLE_CHUNKED_PREFILL=true
unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_ENFORCE_EAGER
unset VLLM_DISABLE_CHUNKED_PREFILL
export VLLM_GPU_MEMORY_UTILIZATION=0.9

set -a
source .env
set +a

MODEL_DIR="${VLLM_MODEL_PATH}"
PORT="${VLLM_PORT:-11002}"
MAX_TOKENS="${VLLM_MAX_TOKENS:-4096}"
SCRIPT_PATH=${SCRIPT_PATH:-run_vllm_server.py}

echo "Launching vLLM inference server on port $PORT..."
python $SCRIPT_PATH \
  --model_path $MODEL_DIR \
  --port $PORT \
  --tensor_parallel_size 4 \
  --max_tokens $MAX_TOKENS
