#!/bin/bash
module load jq zstd pigz parallel libnvidia-container enroot   

CONTAINER_NAME="mingpt"

export HF_CACHE=/tmp
export TRANSFORMERS_CACHE=/tmp
export HF_DATASETS_CACHE=/tmp

# Check if container already exists using enroot list
if ! enroot list | grep -q "^${CONTAINER_NAME}\$"; then
    enroot create --force --name $CONTAINER_NAME ${HOME}/sqsh_files/mingpt.sqsh
fi                

#    --mount ${HOME}/hf_models:/app/hf_models \

enroot start \
       --mount .:/app \
       --mount /lustre/scratch/usr/dw87:/app/data \
       mingpt \
       bash # the name of the command INSIDE THE CONTAINER that you want to run