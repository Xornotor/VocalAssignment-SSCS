#!/bin/bash
docker run -p 8888:8888 -p 6006:6006 -it --rm \
    -v ./app/:/app/VocalAssignment-SSCS \
    --gpus all \
    tf:v1