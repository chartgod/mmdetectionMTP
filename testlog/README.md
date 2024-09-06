# Change Detection Project for Geographic Categories

This repository contains the code and configuration files used to test four deep learning models (Snunet, Changeformer, Changer, and MTP) on six different categories: **Buildings**, **Roads**, **Green Spaces**, **Wildfire Damage**, **Water Bodies**, and **Rivers**.

## Models
The following models are used for change detection tasks:
- **Snunet**
- **Changeformer**
- **Changer**
- **MTP**

Each model is tested across the six categories. The command lines used for each category and model are as follows:

## Testing Commands by Category

### 1. Buildings

- **Snunet**
    ```bash
    CUDA_VISIBLE_DEVICES=1 python /home/lsh/share/CD/open-cd/tools/test.py /home/lsh/share/CD/open-cd/configs/snunet/snunet_c16_256x256_40k_levircd건물.py /home/lsh/share/CD/open-cd/test/건물/snunet/best_mIoU_iter_40000.pth --show-dir /home/lsh/share/CD/open-cd/test/과년도건물snunet > /home/lsh/share/CD/open-cd/test/과년도/건물snunet.txt 2>&1
    ```

- **Changeformer**
    ```bash
    CUDA_VISIBLE_DEVICES=1 python /home/lsh/share/CD/open-cd/tools/test.py /home/lsh/share/CD/open-cd/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd건물.py /home/lsh/share/CD/open-cd/test/건물/changeformer/best_mIoU_iter_40000.pth --show-dir /home/lsh/share/CD/open-cd/test/과년도건물changeformer > /home/lsh/share/CD/open-cd/test/과년도/건물changeformer.txt 2>&1
    ```

- **Changer**
    ```bash
    CUDA_VISIBLE_DEVICES=2 python /home/lsh/share/CD/open-cd/tools/test.py /home/lsh/share/CD/open-cd/configs/changer/changer_ex_r18_512x512_40k_levircd건물.py /home/lsh/share/CD/open-cd/test/건물/changer/best_mIoU_iter_28000.pth --show-dir /home/lsh/share/CD/open-cd/test/과년도건물changer > /home/lsh/share/CD/open-cd/test/과년도/건물changer.txt 2>&1
    ```

- **MTP**
    ```bash
    CUDA_VISIBLE_DEVICES=1 python -u /home/lsh/share/mmsatellite/test.py /home/lsh/share/mmsatellite/configs/mtp/rvsa-l-unet-256-mae-mtp_levir건물.py /home/lsh/share/CD/open-cd/test/건물/mtp/best_mIoU_epoch_150.pth --work-dir=/home/lsh/share/CD/open-cd/test/과년도mtp/건물/ --show-dir=/home/lsh/share/CD/open-cd/test/과년도건물mtp --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
    ```

### 2. Roads

- **Snunet**
    ```bash
    CUDA_VISIBLE_DEVICES=1 python /home/lsh/share/CD/open-cd/tools/test.py /home/lsh/share/CD/open-cd/configs/snunet/snunet_c16_256x256_40k_levircd도로.py /home/lsh/share/CD/open-cd/test/도로/snunet/best_mIoU_iter_40000.pth --show-dir /home/lsh/share/CD/open-cd/test/과년도도로snunet > /home/lsh/share/CD/open-cd/test/과년도/도로snunet.txt 2>&1
    ```

...

### 6. Water Bodies

- **Snunet**
    ```bash
    CUDA_VISIBLE_DEVICES=1 python /home/lsh/share/CD/open-cd/tools/test.py /home/lsh/share/CD/open-cd/configs/snunet/snunet_c16_256x256_40k_levircd수계.py /home/lsh/share/CD/open-cd/test/수계/snunet/best_mIoU_iter_40000.pth --show-dir /home/lsh/share/CD/open-cd/test/과년도수계snunet > /home/lsh/share/CD/open-cd/test/과년도/수계snunet.txt 2>&1
    ```

- **Changeformer**
    ```bash
    CUDA_VISIBLE_DEVICES=5 python /home/lsh/share/CD/open-cd/tools/test.py /home/lsh/share/CD/open-cd/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd수계.py /home/lsh/share/CD/open-cd/test/수계/changeformer/best_mIoU_iter_40000.pth --show-dir /home/lsh/share/CD/open-cd/test/과년도수계changeformer > /home/lsh/share/CD/open-cd/test/과년도/수계changeformer.txt 2>&1
    ```

- **Changer**
    ```bash
    CUDA_VISIBLE_DEVICES=3 python /home/lsh/share/CD/open-cd/tools/test.py /home/lsh/share/CD/open-cd/configs/changer/changer_ex_r18_512x512_40k_levircd수계.py /home/lsh/share/CD/open-cd/test/수계/changer/best_mIoU_iter_40000.pth --show-dir /home/lsh/share/CD/open-cd/test/과년도수계changer > /home/lsh/share/CD/open-cd/test/과년도/수계changer.txt 2>&1
    ```

- **MTP**
    ```bash
    CUDA_VISIBLE_DEVICES=5 python -u /home/lsh/share/mmsatellite/test.py /home/lsh/share/mmsatellite/configs/mtp/rvsa-l-unet-256-mae-mtp_levir수계.py /home/lsh/share/CD/open-cd/test/수계/mtp/best_mIoU_epoch_150.pth --work-dir=/home/lsh/share/CD/open-cd/test/과년도mtp/수계/ --show-dir=/home/lsh/share/CD/open-cd/test/과년도수계mtp --cfg-options val_cfg=None val_dataloader=None val_evaluator=None
    ```

...

## Directory Structure
```bash
/home/lsh/share/CD/open-cd/
├── configs/
│   ├── snunet/
│   ├── changeformer/
│   ├── changer/
│   └── mtp/
├── test/
│   ├── 건물/
│   ├── 도로/
│   ├── 녹지/
│   ├── 산불피해/
│   └── 수계/
