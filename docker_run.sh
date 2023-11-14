#!/bin/bash

docker run -it -p 8888:8888 --name davis_mingpt --gpus all \
    --mount type=bind,source=/mnt/pccfs2/backed_up/davisforster/minGPT_proj2,target=/app/minGPT_proj2 \
    --mount type=bind,source=/mnt/pccfs2/backed_up/davisforster/lm-evaluation-harness,target=/app/lm_eval \
    davis_mingpt