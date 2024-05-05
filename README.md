# Unoffical implementation of Dynamic Fusion
This repo is not an official implementation of the paper [DynamicFusion: Reconstruction and Tracking of Non-Rigid Scenes in Real-Time (CVPR 2015)](https://www.microsoft.com/en-us/research/wp-content/uploads/2015/05/DynamicFusion.pdf). The code is a work in progress and is not yet complete. The goal is to implement the core functionalities of the paper and to provide a simple and easy to understand codebase for learning purposes.
The tree stucture is hard to follow and I haven't find a way to bring it to pytorch.


## Requirements
The core functionalities were implemented in PyTorch (1.10). Open3D (0.14.0) is used for visualisation. Other important dependancies include:

* numpy==1.21.2
* opencv-python==4.5.5
* imageio==2.14.1
* scikit-image==0.19.1
* trimesh==3.9.43

You can create an anaconda environment called `kinfu` with the required dependencies by running:
```
conda env create -f environment.yml
conda activate kinfu
```



## Acknowledgement
This code is heavily dependent on [JingwenWang95's KinectFusion implementation](https://github.com/JingwenWang95/KinectFusion). 

## References
 * [KinectFusion: Real-Time Dense Surface Mapping and Tracking (ISMAR 2011)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf)
 * [Deep Probabilistic Feature-metric Tracking (RA-L and ICRA 2021 presentation)](https://arxiv.org/pdf/2008.13504.pdf)
 * [Taking a Deeper Look at the Inverse Compositional Algorithm (CVPR 2019)](https://arxiv.org/pdf/1812.06861.pdf)
