# mmdetectionMTP
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white)](https://www.python.org/)

Welcome to the mmdetectionMTP repository! This project is built on top of the MTP environment from [ViTAE-Transformer/MTP](https://github.com/ViTAE-Transformer/MTP).

## Environment Settings 🛠️
Follow these steps to set up the environment and install necessary dependencies for training.

### Step-by-Step Setup Guide

#### 1. Create and Activate Virtual Environment
```bash
conda create -n mtp python=3.8.19
conda install pytorch==1.10.0 torchvision==0.11.0 -c pytorch

# Install OpenMMLab Toolkits as Python packages
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0" or mim install "mmcv==2.0.0rc4"
mim install "mmpretrain>=1.0.0rc7"
pip install "mmsegmentation>=1.2.2"
pip install "mmdet>=3.0.0"

git clone https://github.com/likyoo/open-cd.git
cd open-cd
pip install -v -e .

pip install ftfy
pip install regex
pip install timm

#### 2. 🚀 Usage (Train)
ex) Train - MultiGPU 
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python /home/lsh/share/mmsatellite/train.py --config /home/lsh/share/mmsatellite/configs/mtp/rvsa-l-unet-256-mae-mtp_levir.py --work-dir /home/lsh/share/mmsatellite/rvsa-l-unet-256-mae-mtp_levir_workdir

