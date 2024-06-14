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
```

#### 2. Install OpenMMLab Toolkits as Python Packages
```
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
# mim install "mmcv==2.0.0rc4"
mim install "mmpretrain>=1.0.0rc7"
pip install "mmsegmentation>=1.2.2"
pip install "mmdet>=3.0.0"
```

#### 3. Clone and Install open-cd
```
git clone https://github.com/likyoo/open-cd.git
cd open-cd
pip install -v -e .
pip install ftfy
pip install regex
pip install timm
```

## Usage (Train) 🚀 
```
For training with multiple GPUs, use the following command:
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python /home/lsh/share/mmsatellite/train.py --config /home/lsh/share/mmsatellite/configs/mtp/rvsa-l-unet-256-mae-mtp_levir.py --work-dir /home/lsh/share/mmsatellite/rvsa-l-unet-256-mae-mtp_levir_workdir
```

#### MTP - Change Detection (using Open-CD)
```
Training on WHU using UperNet with a backbone network of MAE + MTP pretrained ViT-L + RVSA:
srun -J opencd -p gpu --gres=dcu:4 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python -u tools/train.py configs/mtp/whu/rvsa-l-unet-256-mae-mtp_whu.py \
--work-dir=/diwang22/work_dir/multitask_pretrain/finetune/Change_Detection/whu/rvsa-l-unet-256-mae-mtp_whu \
--launcher="slurm" --cfg-options 'find_unused_parameters'=True
```


## 🎵 Citation
If you find MTP helpful, please consider giving this repo a ⭐ and citing:

```
@ARTICLE{MTP,
  author={Wang, Di and Zhang, Jing and Xu, Minqiang and Liu, Lin and Wang, Dongsheng and Gao, Erzhong and Han, Chengxi and Guo, Haonan and Du, Bo and Tao, Dacheng and Zhang, Liangpei},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={MTP: Advancing Remote Sensing Foundation Model Via Multi-Task Pretraining}, 
  year={2024},
  volume={},
  number={},
  pages={1-24},
  doi={10.1109/JSTARS.2024.3408154}}

url:https://github.com/ViTAE-Transformer/MTP
```
