#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate env/

export CUDA_VISIBLE_DEVICES=0
python dataset/preprocess.py --config configs/fr1_desk.yaml
python kinfu.py --config configs/fr1_desk.yaml --save_dir reconstruct/fr1_desk
