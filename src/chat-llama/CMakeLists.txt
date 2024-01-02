file(COPY
    chat.sh
    DESTINATION ${MY_RUNTIME_DIR}
)

file(COPY
    data/chat-prompts.txt
    DESTINATION ${MY_RUNTIME_DIR}/data
)
