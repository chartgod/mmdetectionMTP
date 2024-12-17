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
