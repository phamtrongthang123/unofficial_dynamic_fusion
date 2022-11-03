#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate env/

export CUDA_VISIBLE_DEVICES=0
# python dataset/preprocess.py --config configs/fr1_desk.yaml
python dynfu.py --config configs/seq003.yaml --save_dir reconstruct/seq003_noreg_noupdategraph_nofusion_noicp
