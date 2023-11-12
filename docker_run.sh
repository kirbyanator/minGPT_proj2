#!/bin/bash

docker run -it -p 8888:8888 --name mingpt \
    --mount type=bind,source=.,target=/app \
    mingpt