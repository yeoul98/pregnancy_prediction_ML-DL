# Version 3: K-Fold Target Encoding (Best OOF)

이 버전은 **Leakage가 없는 Target Encoding**과 **Stacking** 기법을 도입하여 교차 검증(CV) 성능을 최대로 끌어올린 모델입니다.

## Key Strategies
- **K-Fold Target Encoding**: 범주형 변수의 의미를 수치화하기 위해 Smoothing 기법이 적용된 타겟 인코딩을 각 Fold 내부에서 독립적으로 수행 (Data Leakage 방지)
- **Model Diversity**: 하이퍼파라미터(Seed, Depth 등)를 다르게 한 총 9개의 기본 모델(CatBoost, LGBM, XGB 각 3종) 구성
- **Stacking**: Level-0 모델들의 예측값을 메타 피처로 하여 RidgeClassifier(Meta-Learner)를 통해 최종 결합

## Files
- `pipeline_v3.py`: 전체 학습 코드
- `postprocess_v3.py`: 저장된 .npy 예측값을 활용한 후처리 및 앙상블 코드
- `submission_stacking.csv`: Ridge Stacking 기반 제출 파일
