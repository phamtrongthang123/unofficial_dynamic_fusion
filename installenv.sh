#!/bin/bash
# Oct 29th 2022, this install script works for functorch 0.2.0 and pytorch3d 0.7.1

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
rm -rf env/ 
conda create --prefix env/ python=3.9 -y 
conda env update --prefix env/ --file environment.yml --prune
conda activate env/
pip install -r requirements.txt
conda install -y pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install functorch==0.2.0
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath # need this to install pytorch3d
conda install -y -c bottler nvidiacub
conda install -y pytorch3d -c pytorch3d
python -c "import pytorch3d.utils; import functorch" # test calling
