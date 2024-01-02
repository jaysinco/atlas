file(COPY
    chat.sh
    DESTINATION ${MY_RUNTIME_DIR}
)

file(COPY
    data/chat-prompts.txt
    data/mistral-7b-v0.1.Q4_0.gguf
    DESTINATION ${MY_RUNTIME_DIR}/data
)
