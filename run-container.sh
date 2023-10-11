#!/bin/bash
docker run -p 8888:8888 -p 6006:6006 -it --rm \
    -v ./va/:/app/VocalAssignment-SSCS \
    -v ./mpe/:/app/MultiF0 \
    --gpus all \
    tf:v1