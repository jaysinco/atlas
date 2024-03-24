#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODEL=$SCRIPT_DIR/data/mistral-7b-v0.1.Q4_0.gguf
PROMPT_TEMPLATE=$SCRIPT_DIR/data/chat-prompts.txt
USER_NAME="${USER_NAME:-USER}"
AI_NAME="${AI_NAME:-ChatLLaMa}"

# Adjust to the number of CPU cores you want to use.
N_THREAD=`nproc`
# Number of tokens to predict (made it larger than default because we want a long interaction)
N_PREDICTS="${N_PREDICTS:-2048}"

# Note: you can also override the generation options by specifying them on the command line:
# For example, override the context size by doing: ./chatLLaMa --ctx_size 1024
GEN_OPTIONS="${GEN_OPTIONS:---ctx_size 2048 --temp 0.7 --top_k 40 --top_p 0.5 --repeat_last_n 256 --batch_size 1024 --repeat_penalty 1.17647}"

DATE_TIME=$(date +%H:%M)
DATE_YEAR=$(date +%Y)

PROMPT_FILE=$(mktemp -t llamacpp_prompt.XXXXXXX.txt)

sed -e "s/\[\[USER_NAME\]\]/$USER_NAME/g" \
    -e "s/\[\[AI_NAME\]\]/$AI_NAME/g" \
    -e "s/\[\[DATE_TIME\]\]/$DATE_TIME/g" \
    -e "s/\[\[DATE_YEAR\]\]/$DATE_YEAR/g" \
     $PROMPT_TEMPLATE > $PROMPT_FILE

# shellcheck disable=SC2086 # Intended splitting of GEN_OPTIONS
LD_LIBRARY_PATH=$SCRIPT_DIR \
$SCRIPT_DIR/llama-cpp $GEN_OPTIONS \
  --model "$MODEL" \
  --threads "$N_THREAD" \
  --n_predict "$N_PREDICTS" \
  --color --interactive \
  --file ${PROMPT_FILE} \
  --reverse-prompt "${USER_NAME}:" \
  --in-prefix ' ' \
  --log-enable \
  "$@"
