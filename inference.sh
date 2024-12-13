#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


input_jsonl=/home/v-wenxichen/data/s2s/mt_bench/mt_test/test.jsonl
output_dir=/home/v-wenxichen/exp/multi-round-glm-new

# python -m debugpy --listen 5678 --wait-for-client inference_multi-round.py \
python inference_multi-round.py \
  --input-jsonl $input_jsonl \
  --output-dir $output_dir \
  --model-path THUDM/glm-4-voice-9b \
  --tokenizer-path THUDM/glm-4-voice-tokenizer \
  --flow-path ./glm-4-voice-decoder

# python inference_multi-round.py \
#   --input-jsonl $input_jsonl \
#   --output-dir $output_dir \
#   --model-path THUDM/glm-4-voice-9b \
#   --tokenizer-path THUDM/glm-4-voice-tokenizer \
#   --flow-path ./glm-4-voice-decoder
