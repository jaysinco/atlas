#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
git_root="$(git rev-parse --show-toplevel)"

system_prompt='You are a helpful assistant. 你是一个乐于助人的助手。'
# system_prompt='You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。' # Try this one, if you prefer longer response.

model_path=$git_root/src/core/llama/res/ggml-model-q4_0.gguf
first_instruction=$1

$script_dir/llama -m "$model_path" \
--color -i -c 4096 -t 8 --temp 0.5 --top_k 40 --top_p 0.9 --repeat_penalty 1.1 \
--in-prefix-bos --in-prefix ' [INST] ' --in-suffix ' [/INST]' -p \
"[INST] <<SYS>>
$system_prompt
<</SYS>>

$first_instruction [/INST]"
