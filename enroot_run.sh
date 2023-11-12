#!/bin/bash
module load jq zstd pigz parallel libnvidia-container enroot   

CONTAINER_NAME="mingpt"

# Check if container already exists using enroot list
if ! enroot list | grep -q "^${CONTAINER_NAME}\$"; then
    enroot create --force --name $CONTAINER_NAME ${HOME}/sqsh_files/mingpt.sqsh
fi                

#    --mount ${HOME}/hf_models:/app/hf_models \

enroot start \
       --mount .:/app \
       --mount /lustre/scratch/usr/dw87/pile_data_10.jsonl:/app/data
       mingpt \
       bash # the name of the command INSIDE THE CONTAINER that you want to run