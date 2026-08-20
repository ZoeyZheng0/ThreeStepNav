#!/bin/bash

# API key is read from the OPENAI_API_KEY environment variable (required for GPT models):
#   export OPENAI_API_KEY=<your-key>
: "${OPENAI_API_KEY:?Please set the OPENAI_API_KEY environment variable}"

## Full dataset
episodes_to_load=100
exp_name="r2r_gpt5"
gpu_id=2

# ## Debug
# episodes_to_load=20
# exp_name="debug"
# gpu_id=3

# ## First
# episodes_to_load=1
# exp_name="first"
# gpu_id=3

flag="--exp_name $exp_name
      --exp-config run_OpenNav.yaml
      --llm gpt-5-2025-08-07
      --episodes_to_load $episodes_to_load
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      EVAL.SPLIT val_unseen
      "
CUDA_VISIBLE_DEVICES=$gpu_id python run.py $flag