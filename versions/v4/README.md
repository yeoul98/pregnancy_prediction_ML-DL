# Version 4: Precision Pruning & Optuna Optimization

이 버전은 **피처 노이즈 제거**와 **자동화된 최적화** 프로세스를 구축하는 데 초점을 맞춘 고도화 파이프라인입니다.

## Key Strategies
- **Null Importance Pruning**: 타겟을 50회 이상 셔플하여 실제 피처 중요도가 가짜 중요도 분포의 상위 90% 이상인 경우에만 선택 (최종 18개 피처 선정)
- **Feature Interaction**: 중요도 Top-5 피처 간의 곱셈/나눗셈을 통한 고차 상호작용 피처 생성
- **Heterogeneous Stacking**: 트리 모델 외에 Logistic Regression과 KNN을 추가한 11개 이종 모델 스태킹
- **Optuna Optimization**: 1,000회 이상의 시행을 통해 메타 모델의 하이퍼파라미터와 모델별 가중치를 자동 최적화

## Files
- `pipeline_v4.py`: 전체 파이프라인 실행 코드
- `submission_v4.csv`: 최종 제출 파일
