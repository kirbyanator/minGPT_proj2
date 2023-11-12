#!/bin/bash
module load jq zstd pigz parallel libnvidia-container enroot   

CONTAINER_NAME="min_gpt"

# Check if container already exists using enroot list
if ! enroot list | grep -q "^${CONTAINER_NAME}\$"; then
    enroot create --force --name $CONTAINER_NAME ${HOME}/sqsh_files/llama_ft.sqsh
fi                

#    --mount ${HOME}/hf_models:/app/hf_models \

enroot start \
       --mount .:/app \
       min_gpt \
       bash # the name of the command INSIDE THE CONTAINER that you want to run