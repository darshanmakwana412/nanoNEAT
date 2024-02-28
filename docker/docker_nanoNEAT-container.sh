#!/bin/bash
docker run -it --rm --detach-keys="ctrl-a" --name=nanoNEAT-container --gpus '"device=0"' --ipc=host -p 8888:8888 -v /home/darshan/Projects/nanoNEAT:/workspace -v /home/darshan/Projects/nanoNEAT/jupyter.sh:/workspace/jupyter.sh -u 1000:1000 pytorch:23.02-py3
