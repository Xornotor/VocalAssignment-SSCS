#!/bin/bash
docker run -p 8888:8888 -p 6006:6006 -it --rm \
    -v ./va_holdout/:/app/va_holdout \
    -v ./va_kfold/:/app/va_kfold \
    -v ./mpe/:/app/mpe \
    --gpus all \
    tf:v1