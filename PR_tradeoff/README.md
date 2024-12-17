#!/bin/bash

# Open-CD Test Script with Precision-Recall Curve Generation
# ---------------------------------------------------------
# 이 스크립트는 모델 테스트, 확률 맵 저장, Precision-Recall Curve 생성을 수행합니다.

# === 마크다운 형식의 설명 출력 ===
cat << "EOF"
# Open-CD Test Script with Precision-Recall Curve Generation

## **설명**
이 스크립트는 다음 작업을 수행합니다:

1. **모델 테스트**: 설정 파일과 체크포인트를 사용해 Open-CD 모델을 테스트합니다.
2. **확률 맵 저장**: 모델 결과로 생성된 확률 맵 이미지를 저장합니다.
3. **Precision-Recall Curve 생성**: 저장된 확률 맵과 GT 라벨을 비교하여 Precision-Recall Curve를 생성합니다.

---

## **사용법**
```bash
bash run_opencd.sh <config.py> <checkpoint.pth> <work_dir> <prob_dir> <label_dir>

## ** 실행 소스코드 **
CUDA_VISIBLE_DEVICES=7 python preicision_recall_tradeoff_code.py \
    /home/lsh/share/mmsatellite/configs/mtp/rvsa-l-unet-256-mae-mtp_levir_커스텀10000.py \
    /home/lsh/share/mmsatellite/MTP_trainer/levir_cd_커스텀10000/epoch_150.pth \
    --work-dir /home/lsh/share/mmsatellite/tradeoff 
