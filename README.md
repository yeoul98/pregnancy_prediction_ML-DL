# IVFPredictor-v4

난임 시술 성공 여부 예측을 위한 머신러닝 파이프라인 프로젝트입니다. 
데이터 전처리부터 최첨단 앙상블 기법(Stacking, Optuna, Null Importance)까지의 발전 과정을 담고 있습니다.

## 📁 Repository Structure

- `versions/`: 버전별 소스 코드 및 각 단계의 제출 파일
  - `v1/`: Baseline (LGBM 기초 모델)
  - `v2/`: Feature Engineering & Weighted Ensemble (LB 최고점 버전)
  - `v3/`: K-Fold Target Encoding & Ridge Stacking (OOF 최고점 버전)
  - `v4/`: Null Importance Pruning & Optuna Meta-Model Optimization
- `data/`: 원본 데이터 (train.csv, test.csv, 데이터 명세)
- `results/`: 중간 생성물 (.npy) 및 성능 지표

## 🚀 Performance Summary

| Version | OOF AUC | Strategy | Status |
|---------|---------|----------|--------|
| v1 | 0.7100+ | Baseline | Completed |
| **v2** | **0.7401** | **Domain Feature Engineering** | **Best LB** |
| **v3** | **0.7403** | **K-Fold Target Encoding** | **Best OOF** |
| v4 | 0.7128 | Null Importance & Optuna | Completed |

## 🛠 Features (v2/v3 Core)

- **나이 데이터 수치화**: 범주형 나이를 중앙값으로 변환하여 연속형 변수로 활용
- **시술 효율 지표**: 수정률, 이식률, 저장률 등 도메인 지식 기반 파생 변수
- **K-Fold Target Encoding**: 데이터 누수 없이 범주형 데이터의 의미 정보 추출
- **Stacking Ensemble**: RidgeClassifier 및 Optuna를 이용한 가중치 최적화

## 📝 Final Report
상세 분석 보고서는 [여기](./final_report.md)에서 확인하실 수 있습니다.
