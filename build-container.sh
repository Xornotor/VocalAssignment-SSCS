#!/bin/bash
docker image remove -f tf:v1
docker build -t tf:v1 .