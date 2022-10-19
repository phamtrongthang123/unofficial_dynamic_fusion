#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
rm -rf env/ 
conda create --prefix env/ python=3.8 -y 
conda env update --prefix env/ --file environment.yml --prune
conda activate env/
